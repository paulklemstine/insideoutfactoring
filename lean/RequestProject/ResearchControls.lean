import RequestProject.ProjectiveCollision

set_option autoImplicit false

namespace ResearchControls

/-- Standard integral parametrization of the projective Pythagorean conic.
The iid control in the mixing study samples this map modulo the input. -/
def conicPoint (t : ℤ) : ℤ × ℤ × ℤ :=
  (2 * t, 1 - t ^ 2, 1 + t ^ 2)

/-- The iid control really lies on the same conic as the Berggren treatment. -/
theorem conicPoint_pythagorean (t : ℤ) :
    (conicPoint t).1 ^ 2 + (conicPoint t).2.1 ^ 2 =
      (conicPoint t).2.2 ^ 2 := by
  simp only [conicPoint]
  ring

/-- A residue divisible by a hidden divisor produces a GCD at least as large
as that divisor.  This explains why either control or treatment can expose a
factor while the exact proper-divisor gate remains method-independent. -/
theorem divisor_le_gcd_of_dvd_residue {N r p : ℕ}
    (hpN : p ∣ N) (hpr : p ∣ r) : p ∣ Nat.gcd r N := by
  exact Nat.dvd_gcd hpr hpN

/-- Every accepted residue in the new experiment yields an exact factor and
cofactor decomposition; randomness and scheduling are outside this kernel. -/
theorem accepted_residue_exact_split (N r : ℕ)
    (hlo : 1 < Nat.gcd r N) (hhi : Nat.gcd r N < N) :
    ∃ q, N = Nat.gcd r N * q ∧ 1 < Nat.gcd r N ∧ Nat.gcd r N < N := by
  exact ProjectiveCollision.chart_collision_gcd_certificate N r hlo hhi

end ResearchControls
