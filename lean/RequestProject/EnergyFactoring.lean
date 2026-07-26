import Mathlib

set_option autoImplicit false

namespace EnergyFactoring

/-- A computable divisor Hamiltonian: its energy is the square of the division
remainder. Lean's totalized `% 0` convention also makes the characterization
below valid at zero. -/
def residueEnergy (N d : ℕ) : ℕ := (N % d) ^ 2

/-- The ground states of `residueEnergy` are exactly the (not necessarily
proper) divisors of `N`. -/
theorem residueEnergy_eq_zero_iff (N d : ℕ) :
    residueEnergy N d = 0 ↔ d ∣ N := by
  simp [residueEnergy, Nat.pow_eq_zero, Nat.dvd_iff_mod_eq_zero]

/-- Restricting the candidate interval turns a zero-energy state into an exact
certificate of a nontrivial factor, and conversely. -/
theorem residueEnergy_proper_factor_iff (N d : ℕ) (hd : 1 < d) (hDN : d < N) :
    residueEnergy N d = 0 ↔ (d ∣ N ∧ 1 < d ∧ d < N) := by
  rw [residueEnergy_eq_zero_iff N d]
  constructor
  · intro h
    exact ⟨h, hd, hDN⟩
  · intro ⟨h', _, _⟩
    exact h'

/-- Every proper zero-energy candidate yields an explicit complementary factor. -/
theorem ground_state_yields_decomposition (N d : ℕ) (hd : 1 < d) (hDN : d < N)
    (hE : residueEnergy N d = 0) :
    ∃ q, N = d * q ∧ 1 < d ∧ d < N := by
  have hdiv := residueEnergy_eq_zero_iff N d |>.mp hE
  exact ⟨N / d, (Nat.mul_div_cancel' hdiv).symm, hd, hDN⟩

/-- The factorization energy can be evaluated without knowing a factor: it is
zero precisely when an exact multiplication certificate exists. -/
theorem residueEnergy_zero_iff_multiplication (N d : ℕ) :
    residueEnergy N d = 0 ↔ ∃ q, N = d * q := by
  rw [residueEnergy_eq_zero_iff N d]
  simp [dvd_iff_exists_eq_mul_left, mul_comm]

/-- A difference-of-squares ground state supplies the two multiplication
candidates used in Fermat/QS/NFS-style extraction. -/
theorem difference_square_ground_state (N x y : ℤ)
    (hE : x ^ 2 - y ^ 2 - N = 0) : N = (x - y) * (x + y) := by
  linarith [sq_nonneg x, sq_nonneg y]

/-- At the successful inside-out index, the candidate is exactly the odd
factor itself; the coordinate change does not predict that index. -/
theorem inside_out_success_index (N p k : ℕ)
    (hp : p = 2 * k + 1) (hdiv : p ∣ N) :
    residueEnergy N (2 * k + 1) = 0 := by
  rw [hp] at hdiv
  exact residueEnergy_eq_zero_iff N (2 * k + 1) |>.mpr hdiv

end EnergyFactoring
