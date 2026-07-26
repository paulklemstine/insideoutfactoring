#!/usr/bin/env python3
"""PW-012: periodic Berggren-word smooth-order factoring study.

The treatment and controls are frozen independently of outcomes.  Every lane
raises a group element through the same prime-power ladder and accepts only an
exact proper gcd.  Known factors are used only for post-hoc order diagnostics.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, random, statistics
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
BERGGREN = {
    "U": ((1,-2,2),(2,-1,2),(2,-2,3)),
    "A": ((1,2,2),(2,1,2),(2,2,3)),
    "D": ((-1,2,2),(-2,1,2),(-2,2,3)),
}
# Algebraically diverse, fixed before looking at any factorization outcome.
WORDS = ("U","A","D","UA","UD","AU","AD","DU","DA",
         "UAD","UDA","AUD","ADU","DUA","DAU","UUA","AAD","DDA")
BASES = tuple(range(2, 20))
LUCAS_P = tuple(range(3, 21))


def primes_to(n: int) -> list[int]:
    out=[]
    for x in range(2,n+1):
        if all(x%p for p in out if p*p<=x): out.append(x)
    return out


def ladder(bound: int) -> list[int]:
    out=[]
    for p in primes_to(bound):
        q=p
        while q*p<=bound: q*=p
        out.append(q)
    return out


def mmul(a, b, mod: int, counter: list[int]):
    n=len(a); k=len(b); m=len(b[0]); counter[0]+=n*k*m
    return tuple(tuple(sum(a[i][t]*b[t][j] for t in range(k))%mod
                       for j in range(m)) for i in range(n))


def mpow(a, e: int, mod: int, counter: list[int]):
    r=tuple(tuple(int(i==j) for j in range(len(a))) for i in range(len(a)))
    while e:
        if e&1: r=mmul(r,a,mod,counter)
        e//=2
        if e: a=mmul(a,a,mod,counter)
    return r


def word_matrix(word: str):
    c=[0]; r=((1,0,0),(0,1,0),(0,0,1))
    for ch in word: r=mmul(BERGGREN[ch],r,10**100,c)
    return r


def isolate(n: int, residues: Iterable[int]) -> tuple[int|None,int]:
    rs=[x%n for x in residues]; prod=1
    for x in rs: prod=prod*x%n
    gcds=1; g=math.gcd(prod,n)
    if 1<g<n: return g,gcds
    if g==n:
        for x in rs:
            gcds+=1; d=math.gcd(x,n)
            if 1<d<n: return d,gcds
    return None,gcds


def berggren_lane(n: int, bound: int):
    ops=[0]; gcds=0
    for word in WORDS:
        a=tuple(tuple(x%n for x in row) for row in word_matrix(word))
        for q in ladder(bound): a=mpow(a,q,n,ops)
        d,g=isolate(n,(a[i][j]-int(i==j) for i in range(3) for j in range(3)))
        gcds+=g
        if d: return d,ops[0],gcds,word
    return None,ops[0],gcds,""


def multiplicative_lane(n: int, bound: int):
    ops=0; gcds=0
    qs=ladder(bound)
    for base in BASES:
        x=base%n
        for q in qs:
            # Python's powering is used, but cost is counted as binary scalar
            # modular multiplications under the same square-and-multiply model.
            ops += q.bit_length()-1 + max(0,q.bit_count()-1)
            x=pow(x,q,n)
        d,g=isolate(n,(x-1,)); gcds+=g
        if d: return d,ops,gcds,str(base)
    return None,ops,gcds,""


def lucas_lane(n: int, bound: int):
    ops=[0]; gcds=0
    for p in LUCAS_P:
        a=((p%n,(-1)%n),(1,0))
        for q in ladder(bound): a=mpow(a,q,n,ops)
        d,g=isolate(n,(a[i][j]-int(i==j) for i in range(2) for j in range(2)))
        gcds+=g
        if d: return d,ops[0],gcds,str(p)
    return None,ops[0],gcds,""


def projective_order(matrix, prime: int, cap: int=2_000_000) -> int|None:
    """Exact order of a small matrix for diagnostic labels; never used to split."""
    ident=tuple(tuple(int(i==j) for j in range(len(matrix))) for i in range(len(matrix)))
    a=ident; c=[0]; m=tuple(tuple(x%prime for x in row) for row in matrix)
    for k in range(1,cap+1):
        a=mmul(m,a,prime,c)
        if a==ident: return k
    return None


def smooth_part(n: int, bound: int) -> int:
    r=1
    for p in primes_to(bound):
        while n%p==0: n//=p; r*=p
    return r


def bootstrap_ci(diffs: list[float], seed=12012, reps=10000):
    rng=random.Random(seed); vals=[]
    for _ in range(reps): vals.append(sum(rng.choice(diffs) for _ in diffs)/len(diffs))
    vals.sort(); return vals[int(.025*reps)],vals[int(.975*reps)]


def run(cases_path: Path, output: Path, summary_path: Path, bounds=(100,1000)):
    cases=json.loads(cases_path.read_text()); rows=[]
    lanes=(("berggren",berggren_lane),("p_minus_1",multiplicative_lane),("lucas_torus",lucas_lane))
    for case in cases:
        n=int(case["n"])
        for bound in bounds:
            for name,fn in lanes:
                d,ops,gcds,w=fn(n,bound)
                if d is not None and (not 1<d<n or n%d): raise AssertionError("uncertified split")
                rows.append({"case":case["name"],"n":n,"bits":n.bit_length(),"bound":bound,
                             "method":name,"factor":d or "","success":int(d is not None),
                             "modular_multiplications":ops,"gcds":gcds,"witness":w})
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    summary={"protocol":{"cases":len(cases),"bounds":list(bounds),"words":list(WORDS),
                         "bases":list(BASES),"lucas_P":list(LUCAS_P)},"results":{}}
    for bound in bounds:
        by={m:[r for r in rows if r["bound"]==bound and r["method"]==m] for m,_ in lanes}
        success_sets={m:{r["case"] for r in rs if r["success"]} for m,rs in by.items()}
        for m,rs in by.items():
            splits=sum(r["success"] for r in rs); total_ops=sum(r["modular_multiplications"] for r in rs)
            summary["results"][f"{m}_B{bound}"]={"splits":splits,
                "total_modular_multiplications":total_ops,
                "splits_per_million_modular_multiplications":splits*1_000_000/max(1,total_ops),
                "median_modular_multiplications":statistics.median(r["modular_multiplications"] for r in rs)}
        controls=success_sets["p_minus_1"] | success_sets["lucas_torus"]
        summary["results"][f"coverage_B{bound}"]={
            "control_union_splits":len(controls),
            "all_methods_union_splits":len(controls | success_sets["berggren"]),
            "berggren_splits_not_seen_by_controls":len(success_sets["berggren"]-controls)}
        # paired success differences against each established order control
        for ctl in ("p_minus_1","lucas_torus"):
            dif=[a["success"]-b["success"] for a,b in zip(by["berggren"],by[ctl])]
            summary["results"][f"berggren_minus_{ctl}_B{bound}"]={"mean":sum(dif)/len(dif),"bootstrap_95":bootstrap_ci(dif)}
    summary["csv_sha256"]=hashlib.sha256(output.read_bytes()).hexdigest()
    summary_path.write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--cases",type=Path,default=ROOT/"experiments/balanced_independent_cases.json")
    ap.add_argument("--output",type=Path,default=ROOT/"experiments/periodic_word_order_results.csv")
    ap.add_argument("--summary",type=Path,default=ROOT/"experiments/periodic_word_order_summary.json")
    ap.add_argument("--bounds",type=int,nargs="+",default=[100,1000]); args=ap.parse_args()
    run(args.cases,args.output,args.summary,tuple(args.bounds))
