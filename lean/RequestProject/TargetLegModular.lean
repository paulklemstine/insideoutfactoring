import RequestProject.InsideOut

set_option autoImplicit false

namespace PythagoreanDescent

/-- Setting a target coordinate equal to `N` modulo `N` is exactly the
coordinate-zero goal used by inside-out search. -/
theorem target_N_mod_iff_zero (N x : ℤ) :
    Int.ModEq N x N ↔ Int.ModEq N x 0 := by
  simp [Int.modEq_iff_dvd]

/-- With first leg `N`, the Pythagorean equation is equivalently the product
relation `(c-b)(c+b)=N²`.  This form continues to make sense modulo any
modulus. -/
theorem fixed_leg_product_mod (m N b c : ℤ)
    (h : N ^ 2 + b ^ 2 = c ^ 2) :
    Int.ModEq m ((c - b) * (c + b)) (N ^ 2) := by
  have h1 : (c - b) * (c + b) = N ^ 2 := by linear_combination -h
  rw [h1]

/-- Modulo a divisor of `N`, every fixed-leg completion lies on the degenerate
conic `(c-b)(c+b)=0`. -/
theorem fixed_leg_degenerates_mod_divisor (p N b c : ℤ)
    (hpN : p ∣ N) (h : N ^ 2 + b ^ 2 = c ^ 2) :
    Int.ModEq p ((c - b) * (c + b)) 0 := by
  have h1 : (c - b) * (c + b) = N ^ 2 := by linear_combination -h
  have hpN2 : p ∣ N ^ 2 := dvd_pow hpN (by norm_num : (2 : ℕ) ≠ 0)
  rw [Int.ModEq, h1, Int.emod_eq_zero_of_dvd hpN2, Int.zero_emod]

/-- Division-free parametrization of the fixed-leg conic.  Over an odd field,
one may recover the usual coordinates by dividing the last two entries by 2. -/
theorem fixed_leg_modular_parametrization (N u v : ℤ)
    (huv : u * v = N ^ 2) :
    (2 * N) ^ 2 + (v - u) ^ 2 = (v + u) ^ 2 := by
  linear_combination -4 * huv

/-- The same parametrization remains valid after reduction modulo an arbitrary
modulus, which is the form used for public-prime experiments. -/
theorem fixed_leg_modular_parametrization_mod (m N u v : ℤ)
    (huv : Int.ModEq m (u * v) (N ^ 2)) :
    Int.ModEq m ((2 * N) ^ 2 + (v - u) ^ 2) ((v + u) ^ 2) := by
  have h : (v + u) ^ 2 - ((2 * N) ^ 2 + (v - u) ^ 2) = 4 * (u * v - N ^ 2) := by ring
  rw [Int.modEq_iff_dvd] at huv ⊢
  have huv' : m ∣ u * v - N ^ 2 := by
    simpa using huv.neg_right
  exact h.symm ▸ huv'.mul_left 4

end PythagoreanDescent

namespace AdaptivePortfolio

/-- A shifted target-leg hit is accepted only through the ordinary exact GCD
certificate.  The theorem is deliberately independent of how the candidate
coordinate was generated. -/
theorem targetLeg_gcd_certificate (N delta : ℕ)
    (hlo : 1 < Nat.gcd delta N) (hhi : Nat.gcd delta N < N) :
    1 < Nat.gcd delta N ∧ Nat.gcd delta N < N ∧ Nat.gcd delta N ∣ N ∧
      N / Nat.gcd delta N * Nat.gcd delta N = N := by
  refine ⟨hlo, hhi, Nat.gcd_dvd_right delta N, ?_⟩
  exact Nat.div_mul_cancel (Nat.gcd_dvd_right delta N)

end AdaptivePortfolio
