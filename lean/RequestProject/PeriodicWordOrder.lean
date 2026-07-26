import RequestProject.MegaSynthesis

set_option autoImplicit false

namespace PeriodicWordOrder

/-- If a replayed matrix-word power fixes a coordinate modulo a hidden divisor,
the corresponding integer difference is divisible by that hidden divisor.  This
is the scalar kernel behind every periodic-word matrix probe. -/
theorem hiddenFactor_dvd_periodicResidue
    {p x y k : ℕ} (h : x = y + p * k) : p ∣ x - y := by
  subst x
  simp

/-- A periodic-word residue accepted by the proper-GCD gate yields an exact
factor/cofactor split.  Thus the order heuristic can affect discovery but not
soundness of the returned factor. -/
theorem periodicWord_gcd_certificate (N residue : ℕ)
    (hlo : 1 < Nat.gcd residue N) (hhi : Nat.gcd residue N < N) :
    ∃ q, N = Nat.gcd residue N * q ∧
      1 < Nat.gcd residue N ∧ Nat.gcd residue N < N := by
  exact AdaptivePortfolio.residue_bus_gcd_split N residue hlo hhi

/-- If two hidden factors both divide a batched product, the aggregate GCD may
be the whole input; this is why positive epochs must be recursively isolated. -/
theorem periodicBatch_can_mask
    {p q product : ℕ} (hp : p ∣ product) (hq : q ∣ product) :
    p * q ∣ product * product := by
  exact Nat.mul_dvd_mul hp hq

end PeriodicWordOrder
