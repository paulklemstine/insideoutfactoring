import RequestProject.ThinAscension
import RequestProject.AdaptivePortfolio

set_option autoImplicit false

namespace PythagoreanDescent

/-- Pair a goal covector with a tree state. -/
def incidence (ell v : Triple) : ℤ := ell.a * v.a + ell.b * v.b + ell.c * v.c

/-- Pull a coordinate goal backward through a branch.  This is multiplication
by the transpose of the corresponding Berggren child matrix. -/
def pullback : Branch → Triple → Triple
  | .U, ell =>
      ⟨ell.a + 2 * ell.b + 2 * ell.c,
       -2 * ell.a - ell.b - 2 * ell.c,
       2 * ell.a + 2 * ell.b + 3 * ell.c⟩
  | .A, ell =>
      ⟨ell.a + 2 * ell.b + 2 * ell.c,
       2 * ell.a + ell.b + 2 * ell.c,
       2 * ell.a + 2 * ell.b + 3 * ell.c⟩
  | .D, ell =>
      ⟨-ell.a - 2 * ell.b - 2 * ell.c,
       2 * ell.a + ell.b + 2 * ell.c,
       2 * ell.a + 2 * ell.b + 3 * ell.c⟩

/-- Goal-to-start duality: pulling a goal backward gives exactly the same
incidence as executing the state one step forward. -/
theorem incidence_pullback_child (s : Branch) (ell v : Triple) :
    incidence (pullback s ell) v = incidence ell (child s v) := by
  cases s <;> simp [pullback, child, childU, childA, childD, incidence] <;> ring_nf

/-- The same duality remains valid after reduction modulo the target. -/
theorem incidence_pullback_child_modEq (N : ℤ) (s : Branch) (ell v : Triple) :
    Int.ModEq N (incidence (pullback s ell) v) (incidence ell (child s v)) := by
  exact Int.modEq_of_dvd (by simp [incidence_pullback_child])

end PythagoreanDescent

namespace AdaptivePortfolio

/-- A meet-in-the-middle incidence is safe to accept precisely when its GCD
with `N` passes the same strict proper-factor gate as every other proposal. -/
theorem insideOut_incidence_certificate (N x : ℕ)
    (hlo : 1 < Nat.gcd x N) (hhi : Nat.gcd x N < N) :
    1 < Nat.gcd x N ∧ Nat.gcd x N < N ∧ Nat.gcd x N ∣ N ∧
      N / Nat.gcd x N * Nat.gcd x N = N := by
  exact ⟨hlo, hhi, Nat.gcd_dvd_right _ _, Nat.div_mul_cancel (Nat.gcd_dvd_right _ _)⟩

end AdaptivePortfolio
