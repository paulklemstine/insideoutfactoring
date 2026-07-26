"""AQ-010: Auxiliary-Prime Orbit Features Predict Factor Yield.

Research question: Can auxiliary-prime orbit features predict whether a longer walk
will yield a factor via chart determinant + GCD?

Redesign after debugging:
- Use SHORTER semiprimes where sqrt(p) collisions ARE reachable in limited steps
- Generate many (state, longer_future_label) pairs
- Features at step t, label = whether gcd is proper factor at step t+delta
- Train on small N (reachably positive), test on larger N

Key: we need enough positive samples to train a classifier.
"""
from __future__ import annotations
import random
import sys
from collections import defaultdict
from math import gcd
from typing import NamedTuple

# ------------------------------------------------------------------
# Auxiliary primes
# ------------------------------------------------------------------
AUX_PRIMES = [3, 5, 7, 11, 13, 17]

# ------------------------------------------------------------------
# Triple and chart determinant
# ------------------------------------------------------------------

class Triple(NamedTuple):
    a: int
    b: int
    c: int

    def __repr__(self):
        return f"({self.a}, {self.b}, {self.c})"

def chart_det(t1: Triple, t2: Triple, N: int) -> int:
    """Chart determinant: a2*(c1+b1) - a1*(c2+b2) mod N."""
    a1, b1, c1 = t1.a % N, t1.b % N, t1.c % N
    a2, b2, c2 = t2.a % N, t2.b % N, t2.c % N
    return a2 * (c1 + b1) - a1 * (c2 + b2)

def apply_U(t: Triple, N: int) -> Triple:
    a, b, c = t.a % N, t.b % N, t.c % N
    return Triple(
        (a + 2*b - 2*c) % N,
        (-2*a - b + 2*c) % N,
        (-2*a - 2*b + 3*c) % N
    )
def apply_A(t: Triple, N: int) -> Triple:
    a, b, c = t.a % N, t.b % N, t.c % N
    return Triple(
        (a + 2*b - 2*c) % N,
        (2*a + b - 2*c) % N,
        (-2*a - 2*b + 3*c) % N
    )
def apply_D(t: Triple, N: int) -> Triple:
    a, b, c = t.a % N, t.b % N, t.c % N
    return Triple(
        (-a - 2*b + 2*c) % N,
        (2*a + b - 2*c) % N,
        (-2*a - 2*b + 3*c) % N
    )

BRANCHES = {'U': apply_U, 'A': apply_A, 'D': apply_D}
BRANCH_KEYS = list(BRANCHES.keys())

# ------------------------------------------------------------------
# Feature extraction
# ------------------------------------------------------------------

def feats(t: Triple, N: int):
    """60 features: a,b,c,tr,c+b,c-b mod each aux prime."""
    a, b, c = t.a % N, t.b % N, t.c % N
    tr = (a + c) % N
    cp = (c + b) % N
    cm = (c - b) % N
    out = []
    for p in AUX_PRIMES:
        out.extend([a % p, b % p, c % p, tr % p, cp % p, cm % p])
    return out

# ------------------------------------------------------------------
# Data generation
# ------------------------------------------------------------------

def gen_data(bits: int, num_pairs: int, walk_len: int, seed: int):
    """Generate orbit pair data with labels.

    Returns list of (features, label, N, bits, step) where:
    - features = (f1, f2) concatenated from both endpoints at step t
    - label = 1 if gcd(chart_det(t1,t2), N) is a proper factor at ANY step t' > t
      within the next delta=5 steps
    This is a forward-looking label: "will this state lead to a factor?"
    """
    # Generate semiprime
    from sympy import nextprime
    p0 = nextprime(2 ** (bits // 2))
    q0 = nextprime(p0 + 2)
    N = p0 * q0

    rng1 = random.Random(seed)
    rng2 = random.Random(seed + 77777)

    t1 = Triple(3 % N, 4 % N, 5 % N)
    t2 = Triple(7 % N, 11 % N, 13 % N)

    s1 = []; s2 = []
    data = []
    delta = 5

    for step in range(walk_len):
        b1 = BRANCH_KEYS[rng1.randint(0, 2)]
        t1 = BRANCHES[b1](t1, N)
        s1.append(b1)
        b2 = BRANCH_KEYS[rng2.randint(0, 2)]
        t2 = BRANCHES[b2](t2, N)
        s2.append(b2)

        det = chart_det(t1, t2, N)
        g_now = gcd(abs(det), N)

        # Forward label: does any step in next delta give a factor?
        future_hit = 0
        for d in range(1, delta + 1):
            if step + d >= walk_len:
                break
            # Simulate d more steps
            tt1, tt2 = t1, t2
            for _ in range(d):
                bb1 = BRANCH_KEYS[rng1.randint(0, 2)]
                tt1 = BRANCHES[bb1](tt1, N)
                bb2 = BRANCH_KEYS[rng2.randint(0, 2)]
                tt2 = BRANCHES[bb2](tt2, N)
            dd = chart_det(tt1, tt2, N)
            gg = gcd(abs(dd), N)
            if 1 < gg < N:
                future_hit = 1
                break

        label = future_hit
        f = feats(t1, N) + feats(t2, N)
        data.append((f, label, N, bits, step))

    return data

# ------------------------------------------------------------------
# Simple Decision Tree
# ------------------------------------------------------------------

class SimpleDT:
    def __init__(self, max_depth=6, min_leaf=10):
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.root = None

    def _gini(self, labels):
        from collections import Counter
        c = Counter(labels)
        n = len(labels)
        if n == 0:
            return 0.0
        g = 1.0
        for v in c.values():
            p = v / n
            g -= p * p
        return g

    def _split_ig(self, X, y, fi, thresh):
        left_i = [i for i, x in enumerate(X) if x[fi] <= thresh]
        right_i = [i for i, x in enumerate(X) if x[fi] > thresh]
        if not left_i or not right_i:
            return -1, None, None
        left_y = [y[i] for i in left_i]
        right_y = [y[i] for i in right_i]
        n = len(y)
        parent_gini = self._gini(y)
        child_gini = (len(left_y) * self._gini(left_y) + len(right_y) * self._gini(right_y)) / n
        return parent_gini - child_gini, left_i, right_i

    def _build(self, X, y, depth):
        from collections import Counter
        if depth >= self.max_depth or len(y) < self.min_leaf:
            c = Counter(y)
            return c.most_common(1)[0][0]

        best_ig = -1
        best = None
        n_fi = min(12, len(X[0]))
        fi_list = random.sample(range(len(X[0])), n_fi)

        for fi in fi_list:
            vals = sorted(set(x[fi] for x in X))
            for thresh in [vals[len(vals)//4], vals[len(vals)//2]]:
                ig, li, ri = self._split_ig(X, y, fi, thresh)
                if ig > best_ig:
                    best_ig = ig
                    best = (fi, thresh, li, ri)

        if best is None or best_ig <= 0:
            c = Counter(y)
            return c.most_common(1)[0][0]

        fi, thresh, left_i, right_i = best
        left_X = [X[i] for i in left_i]
        left_y = [y[i] for i in left_i]
        right_X = [X[i] for i in right_i]
        right_y = [y[i] for i in right_i]

        return {
            'fi': fi, 'thresh': thresh,
            'left': self._build(left_X, left_y, depth + 1),
            'right': self._build(right_X, right_y, depth + 1),
        }

    def fit(self, X, y):
        random.seed(42)
        self.root = self._build(X, y, 0)

    def _pred_one(self, x, node):
        if isinstance(node, dict):
            if x[node['fi']] <= node['thresh']:
                return self._pred_one(x, node['left'])
            return self._pred_one(x, node['right'])
        return node

    def predict(self, X):
        return [self._pred_one(x, self.root) for x in X]

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def run():
    print("=" * 60)
    print("AQ-010: Auxiliary-Prime Orbit Features Predict Factor Yield")
    print("=" * 60)

    # Design: use BIT SIZES where collisions are reachable
    # For p≈2^16=65536, sqrt(p)=256 → need walk_len >= 300
    # For p≈2^10=1024, sqrt(p)=32 → walk_len=40 is enough
    TRAIN_BITS = [10, 12, 14, 16, 18]
    TEST_BITS = [20, 22, 24]
    WALK_LEN = 80       # enough for sqrt(p) at 18 bits
    NUM_PAIRS = 500     # orbit pairs per semiprime
    NUM_SEMIS = 5       # semiprimes per bit size

    print(f"\nDesign: train on {TRAIN_BITS}, test on {TEST_BITS}")
    print(f"Walk len={WALK_LEN}, pairs={NUM_PAIRS}, semis per bits={NUM_SEMIS}")
    print(f"Features: 2 × 6 × {len(AUX_PRIMES)} = {2*6*len(AUX_PRIMES)}")

    all_train, all_test = [], []
    for bits in TRAIN_BITS:
        for si in range(NUM_SEMIS):
            d = gen_data(bits, NUM_PAIRS, WALK_LEN, seed=si * 1000 + bits)
            all_train.extend(d)
    for bits in TEST_BITS:
        for si in range(NUM_SEMIS):
            d = gen_data(bits, NUM_PAIRS, WALK_LEN, seed=si * 2000 + bits)
            all_test.extend(d)

    X_train = [f for f, l, *_ in all_train]
    y_train = [l for f, l, *_ in all_train]
    X_test = [f for f, l, *_ in all_test]
    y_test = [l for f, l, *_ in all_test]

    pos_rate_test = sum(y_test) / len(y_test)
    random_baseline = 1 - pos_rate_test

    print(f"\nTrain: {len(X_train)} samples, {sum(y_train)} pos ({100*sum(y_train)/len(X_train):.1f}%)")
    print(f"Test:  {len(X_test)} samples, {sum(y_test)} pos ({100*pos_rate_test:.1f}%)")
    print(f"Random baseline acc: {random_baseline:.3f}")

    clf = SimpleDT(max_depth=8, min_leaf=15)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    tp = sum(1 for p, t in zip(y_pred, y_test) if p == 1 and t == 1)
    tn = sum(1 for p, t in zip(y_pred, y_test) if p == 0 and t == 0)
    fp = sum(1 for p, t in zip(y_pred, y_test) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(y_pred, y_test) if p == 0 and t == 1)

    acc = (tp + tn) / len(y_test)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    lift = acc - random_baseline

    y_pred_train = clf.predict(X_train)
    train_acc = sum(p == t for p, t in zip(y_pred_train, y_train)) / len(y_train)

    print(f"\nDecision Tree:")
    print(f"  Train acc: {train_acc:.3f}")
    print(f"  Test acc:  {acc:.3f} (baseline={random_baseline:.3f})")
    print(f"  Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}")
    print(f"  TP={tp}, TN={tn}, FP={fp}, FN={fn}")

    # Per-test-bits breakdown
    print(f"\nPer test-bits:")
    for bits in TEST_BITS:
        bd = [(f, l) for f, l, N, b, *_ in all_test if b == bits]
        Xb = [f for f, l in bd]
        yb = [l for f, l in bd]
        ypb = clf.predict(Xb)
        accb = sum(p == t for p, t in zip(ypb, yb)) / len(yb)
        posb = sum(yb)
        print(f"  {bits:3d} bits: acc={accb:.3f}, pos_rate={posb/len(yb):.3f}, n={len(yb)}")

    print(f"\n{'='*60}")
    if acc > random_baseline + 0.05 and prec > 0.1:
        decision = "PROMOTED: features predict factor yield"
    elif acc > random_baseline + 0.01:
        decision = "WEAK POSITIVE: marginal lift"
    else:
        decision = "REJECTED: no predictive lift"

    print(f"Decision: {decision}")
    print(f"Lift: {lift:+.3f} over random")
    print(f"{'='*60}")

    return dict(decision=decision, acc=acc, baseline=random_baseline,
                lift=lift, prec=prec, rec=rec, f1=f1,
                n_train=len(X_train), n_test=len(X_test))


if __name__ == '__main__':
    run()
