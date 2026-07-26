import RequestProject.ResearchControls

set_option autoImplicit false

namespace ProductTreeBatching

/-- If two coprime hidden divisors occur in different residues of one epoch,
their product divides both the epoch product and the input.  Thus a full-input
GCD can mask the separate factors and must be recursively isolated. -/
theorem coprime_factors_mask_in_epoch {N x y p q : ℕ}
    (hpq : Nat.Coprime p q) (hpN : p ∣ N) (hqN : q ∣ N)
    (hpx : p ∣ x) (hqy : q ∣ y) :
    p * q ∣ Nat.gcd (x * y) N := by
  apply Nat.dvd_gcd
  · -- p * q ∣ x * y
    apply Nat.Coprime.mul_dvd_of_dvd_of_dvd hpq
    · exact dvd_mul_of_dvd_left hpx y
    · exact dvd_mul_of_dvd_right hqy x
  · -- p * q ∣ N
    exact Nat.Coprime.mul_dvd_of_dvd_of_dvd hpq hpN hqN

/-- Once recursive isolation produces a proper epoch GCD, it gives the same
exact factor/cofactor certificate as every other worker in the laboratory. -/
theorem isolated_epoch_exact_split (N product : ℕ)
    (hlo : 1 < Nat.gcd product N) (hhi : Nat.gcd product N < N) :
    ∃ q, N = Nat.gcd product N * q ∧
      1 < Nat.gcd product N ∧ Nat.gcd product N < N := by
  have hdvd := Nat.gcd_dvd_right product N
  rcases hdvd with ⟨q, hq⟩
  exact ⟨q, hq, hlo, hhi⟩

end ProductTreeBatching
