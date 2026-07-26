import Mathlib

set_option autoImplicit false

namespace PythagoreanDescent

/-- An integer triple, represented as a column vector `(a,b,c)`. -/
structure Triple where
  a : ℤ
  b : ℤ
  c : ℤ
  deriving DecidableEq, Repr

/-- The three Berggren child transforms. -/
def childU (v : Triple) : Triple :=
  ⟨v.a - 2 * v.b + 2 * v.c,
   2 * v.a - v.b + 2 * v.c,
   2 * v.a - 2 * v.b + 3 * v.c⟩

def childA (v : Triple) : Triple :=
  ⟨v.a + 2 * v.b + 2 * v.c,
   2 * v.a + v.b + 2 * v.c,
   2 * v.a + 2 * v.b + 3 * v.c⟩

def childD (v : Triple) : Triple :=
  ⟨-v.a + 2 * v.b + 2 * v.c,
   -2 * v.a + v.b + 2 * v.c,
   -2 * v.a + 2 * v.b + 3 * v.c⟩

/-- The three candidate inverse transforms.  Exactly one is the positive
primitive parent for a non-root primitive Pythagorean triple. -/
def parentU (v : Triple) : Triple :=
  ⟨v.a + 2 * v.b - 2 * v.c,
   -2 * v.a - v.b + 2 * v.c,
   -2 * v.a - 2 * v.b + 3 * v.c⟩

def parentA (v : Triple) : Triple :=
  ⟨v.a + 2 * v.b - 2 * v.c,
   2 * v.a + v.b - 2 * v.c,
   -2 * v.a - 2 * v.b + 3 * v.c⟩

def parentD (v : Triple) : Triple :=
  ⟨-v.a - 2 * v.b + 2 * v.c,
   2 * v.a + v.b - 2 * v.c,
   -2 * v.a - 2 * v.b + 3 * v.c⟩

inductive Branch
  | U | A | D
  deriving DecidableEq, Repr

def child : Branch → Triple → Triple
  | .U => childU
  | .A => childA
  | .D => childD

def parent : Branch → Triple → Triple
  | .U => parentU
  | .A => parentA
  | .D => parentD

/-- Each displayed parent equation really inverts its corresponding child. -/
theorem parent_child (s : Branch) (v : Triple) :
    parent s (child s v) = v := by
  cases s <;> simp [parent, child, childU, parentU, childA, parentA, childD, parentD] <;> ring

/-- The inverse also works in the other order. -/
theorem child_parent (s : Branch) (v : Triple) :
    child s (parent s v) = v := by
  cases s <;> simp only [child, parent, childU, childA, childD, parentU, parentA, parentD]
  all_goals ring

/-- Once the two branch labels are known, the grandparent is obtained by
applying the two inverse equations successively. -/
def grandparent (parentBranch grandparentBranch : Branch) (v : Triple) : Triple :=
  parent grandparentBranch (parent parentBranch v)

/-- Descend repeatedly toward the root using a supplied sequence of branch
labels (the first label describes the current node). -/
def ancestor : List Branch → Triple → Triple
  | [], v => v
  | s :: ss, v => ancestor ss (parent s v)

/-- Euclid parameters determine the entire triple, explaining why two scalar
coordinates suffice even though every node has three children. -/
def fromParams (m n : ℤ) : Triple :=
  ⟨m ^ 2 - n ^ 2, 2 * m * n, m ^ 2 + n ^ 2⟩

/-- The Pythagorean identity in Euclid's two-parameter coordinates. -/
theorem fromParams_pythagorean (m n : ℤ) :
    (fromParams m n).a ^ 2 + (fromParams m n).b ^ 2 =
      (fromParams m n).c ^ 2 := by
  unfold fromParams
  ring

/-- A known Pythagorean triple with leg `a` gives a difference-of-squares
representation of `a²`. -/
theorem pythagorean_difference (a b c : ℤ)
    (h : a ^ 2 + b ^ 2 = c ^ 2) :
    (c - b) * (c + b) = a ^ 2 := by
  linarith [sq_nonneg a, sq_nonneg b, sq_nonneg c]

/-- In the factor-bearing embedding, the apparently geometric coordinates
contain the squared factors explicitly. -/
theorem factor_embedding_decodes (a b c p q : ℤ)
    (ha : a = p * q)
    (hb : 2 * b = q ^ 2 - p ^ 2)
    (hc : 2 * c = q ^ 2 + p ^ 2) :
    c - b = p ^ 2 ∧ c + b = q ^ 2 ∧ a = p * q := by
  constructor
  · linarith
  constructor
  · linarith
  · exact ha

end PythagoreanDescent
