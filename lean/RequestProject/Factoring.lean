import Mathlib

set_option autoImplicit false

namespace BookFactoring

/-- The book's "thin triple" is genuinely Pythagorean (integer form, avoiding division). -/
theorem thin_triple_identity (N : ℤ) :
    (2 * N) ^ 2 + (N ^ 2 - 1) ^ 2 = (N ^ 2 + 1) ^ 2 := by
  ring

/-- The algebraic kernel of Fermat, Pythagorean, and congruence-of-squares factoring. -/
theorem difference_of_squares (x y : ℤ) :
    x ^ 2 - y ^ 2 = (x - y) * (x + y) := by
  ring

/-- A gcd probe always returns a divisor of the target integer. -/
theorem gcd_probe_divides (a N : ℕ) : Nat.gcd a N ∣ N := by
  exact Nat.gcd_dvd_right a N

/-- The standard certificate saying that a gcd probe found a proper factor. -/
theorem gcd_probe_is_nontrivial (a N : ℕ)
    (hlo : 1 < Nat.gcd a N) (hhi : Nat.gcd a N < N) :
    Nat.gcd a N ∣ N ∧ 1 < Nat.gcd a N ∧ Nat.gcd a N < N := by
  exact ⟨Nat.gcd_dvd_right a N, hlo, hhi⟩

/-- A congruence of squares is a multiple of a difference-of-squares product.
This is the algebraic extraction step used by QS and NFS before taking gcds. -/
theorem congruent_squares_product (x y N : ℤ)
    (h : N ∣ x ^ 2 - y ^ 2) : N ∣ (x - y) * (x + y) := by
  rw [← difference_of_squares]
  exact h

/-- Both gcd probes used after a congruence of squares divide the target. -/
theorem congruent_squares_gcd_probes (x y N : ℕ) :
    Nat.gcd (x + y) N ∣ N ∧ Nat.gcd (x - y) N ∣ N := by
  exact ⟨Nat.gcd_dvd_right _ _, Nat.gcd_dvd_right _ _⟩

/-- The IOF step index `k = (p-1)/2` merely enumerates the odd candidate `p = 2k+1`. -/
theorem iof_index_recovers_odd_candidate (p k : ℕ)
    (hp : p = 2 * k + 1) : 2 * k + 1 = p := by
  exact hp.symm

/-- Consequently the advertised IOF gcd at that step is exactly ordinary odd trial division. -/
theorem iof_gcd_is_trial_division (N p k : ℕ)
    (hp : p = 2 * k + 1) : Nat.gcd (2 * k + 1) N = Nat.gcd p N := by
  rw [← hp]

/-- The book's parabolic energy has constant second finite difference eight. -/
theorem parabolic_energy_second_difference (N k : ℤ) :
    (N - 2 * (k + 2)) ^ 2 - 2 * (N - 2 * (k + 1)) ^ 2 +
      (N - 2 * k) ^ 2 = 8 := by
  ring

/-- Every actual divisor is a zero of the natural residue-energy landscape. -/
theorem residue_energy_zero_of_dvd (d N : ℕ) (hd : d ∣ N) :
    (N % d) ^ 2 = 0 := by
  rw [Nat.mod_eq_zero_of_dvd hd]
  norm_num

end BookFactoring
