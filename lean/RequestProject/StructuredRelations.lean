import RequestProject.FibonacciPythagorean
import RequestProject.AuxiliaryOrbit

set_option autoImplicit false

namespace StructuredRelations

open PythagoreanDescent

/-- Every Berggren branch preserves the integral quadratic form defining the
Pythagorean cone.  Thus short tree words provide structured points, but this
identity alone says nothing about their smoothness. -/
theorem child_preserves_pythagorean (s : Branch) (v : Triple)
    (h : v.a ^ 2 + v.b ^ 2 = v.c ^ 2) :
    (child s v).a ^ 2 + (child s v).b ^ 2 = (child s v).c ^ 2 := by
  cases s with
  | U =>
    simp [child, childU]
    linear_combination h
  | A =>
    simp [child, childA]
    linear_combination h
  | D =>
    simp [child, childD]
    linear_combination h

/-- The polynomial sampled by SR-011 is an exact quadratic-sieve relation:
`x² - N` and `x²` agree modulo `N`. -/
theorem qs_relation (N x : ℤ) : Int.ModEq N (x ^ 2 - N) (x ^ 2) := by
  simp [Int.ModEq]

/-- Two square congruences give the standard difference-of-squares GCD
boundary.  A proper GCD is an exact divisor and supplies its cofactor. -/
theorem square_collision_exact_split (N x y : ℕ)
    (hcong : x ^ 2 % N = y ^ 2 % N)
    (hlo : 1 < Nat.gcd (x + y) N) (hhi : Nat.gcd (x + y) N < N) :
    x ^ 2 % N = y ^ 2 % N ∧
    ∃ q, N = Nat.gcd (x + y) N * q ∧ 1 < Nat.gcd (x + y) N ∧
      Nat.gcd (x + y) N < N := by
  refine ⟨hcong, ?_⟩
  have hdiv : Nat.gcd (x + y) N ∣ N := Nat.gcd_dvd_right _ _
  obtain ⟨q, hq⟩ := hdiv
  exact ⟨q, hq, hlo, hhi⟩

end StructuredRelations
