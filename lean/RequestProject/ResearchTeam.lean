import RequestProject.MegaSynthesis

set_option autoImplicit false

namespace FactoringResearch

/-- The three projective-collision residues (the `2 × 2` minors of two
three-coordinate states). -/
def collisionResidues (x y : ℕ × ℕ × ℕ) : List ℕ :=
  [x.1 * y.2.1, x.2.1 * y.1,
   x.1 * y.2.2, x.2.2 * y.1,
   x.2.1 * y.2.2, x.2.2 * y.2.1]

/-- If two triples are proportional modulo `p`, each cross-product difference
is zero modulo `p`.  This is the algebra behind the projective-collision worker. -/
theorem projective_collision_minor_zero
    (p k a b c u v w : ℕ)
    (ha : a % p = (k * u) % p)
    (hb : b % p = (k * v) % p)
    (hc : c % p = (k * w) % p) :
    (a * v) % p = (b * u) % p ∧
    (a * w) % p = (c * u) % p ∧
    (b * w) % p = (c * v) % p := by
  have ha' : a ≡ k * u [MOD p] := ha
  have hb' : b ≡ k * v [MOD p] := hb
  have hc' : c ≡ k * w [MOD p] := hc
  constructor
  · have h1 := ha'.mul (Nat.ModEq.refl v)
    have h2 := hb'.mul (Nat.ModEq.refl u)
    exact h1.trans (by simpa [mul_assoc, mul_left_comm, mul_comm] using h2.symm)
  constructor
  · have h1 := ha'.mul (Nat.ModEq.refl w)
    have h2 := hc'.mul (Nat.ModEq.refl u)
    exact h1.trans (by simpa [mul_assoc, mul_left_comm, mul_comm] using h2.symm)
  · have h1 := hb'.mul (Nat.ModEq.refl w)
    have h2 := hc'.mul (Nat.ModEq.refl v)
    exact h1.trans (by simpa [mul_assoc, mul_left_comm, mul_comm] using h2.symm)

/-- Regardless of how an experimental researcher generated a collision
residue, a nontrivial GCD with `N` passes the same exact factor gate. -/
theorem projective_collision_gcd_certificate (N r : ℕ)
    (hlo : 1 < Nat.gcd r N) (hhi : Nat.gcd r N < N) :
    ∃ q, N = Nat.gcd r N * q ∧ 1 < Nat.gcd r N ∧ Nat.gcd r N < N := by
  exact AdaptivePortfolio.residue_bus_gcd_split N r hlo hhi

end FactoringResearch
