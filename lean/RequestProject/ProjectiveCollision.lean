import RequestProject.ResearchTeam

set_option autoImplicit false

namespace ProjectiveCollision

/-- The two homogeneous determinants obtained from the standard rational charts
of the projective Pythagorean conic. -/
def chartProducts (x y : ℕ × ℕ × ℕ) : List ℕ :=
  [x.1 * (y.2.2 + y.2.1), y.1 * (x.2.2 + x.2.1),
   x.1 * y.2.2 + y.1 * x.2.1,
   x.1 * y.2.1 + y.1 * x.2.2]

/-- Proportional triples collide in both conic charts.  The equalities are
written without natural-number subtraction, exactly as used before taking the
corresponding determinant residues. -/
theorem proportional_implies_chart_collisions
    (p k a b c u v w : ℕ)
    (ha : a % p = (k * u) % p)
    (hb : b % p = (k * v) % p)
    (hc : c % p = (k * w) % p) :
    (a * (w + v)) % p = (u * (c + b)) % p ∧
    (a * w + u * b) % p = (a * v + u * c) % p := by
  have h := FactoringResearch.projective_collision_minor_zero p k a b c u v w ha hb hc
  obtain ⟨h1, h2, h3⟩ := h
  constructor
  · -- (a * (w + v)) % p = (u * (c + b)) % p
    -- a * w ≡ c * u (mod p) and a * v ≡ b * u (mod p)
    -- So a * w + a * v ≡ c * u + b * u = u * (c + b) (mod p)
    rw [mul_add]
    have haw : (a * w) % p = (u * c) % p := by rw [h2]; ring_nf
    have hav : (a * v) % p = (u * b) % p := by rw [h1]; ring_nf
    rw [Nat.add_mod, haw, hav]
    rw [← Nat.add_mod]
    ring_nf
  · -- (a * w + u * b) % p = (a * v + u * c) % p
    -- a * w ≡ c * u (mod p) and a * v ≡ b * u (mod p)
    -- So a * w + u * b ≡ c * u + u * b = u * (b + c) (mod p)
    -- And a * v + u * c ≡ b * u + u * c = u * (b + c) (mod p)
    have haw : (a * w) % p = (u * c) % p := by rw [h2]; ring_nf
    have hav : (a * v) % p = (u * b) % p := by rw [h1]; ring_nf
    -- Both sides equal (u * (b + c)) % p
    have lhs_eq : (a * w + u * b) % p = (u * (b + c)) % p := by
      calc (a * w + u * b) % p = (a * w % p + u * b % p) % p := Nat.add_mod _ _ _
        _ = (u * c % p + u * b % p) % p := by rw [haw]
        _ = (u * (b + c)) % p := by rw [← Nat.add_mod]; ring_nf
    have rhs_eq : (a * v + u * c) % p = (u * (b + c)) % p := by
      calc (a * v + u * c) % p = (a * v % p + u * c % p) % p := Nat.add_mod _ _ _
        _ = (u * b % p + u * c % p) % p := by rw [hav]
        _ = (u * (b + c)) % p := by rw [← Nat.add_mod]; ring_nf
    rw [lhs_eq, rhs_eq]

/-- Every nontrivial GCD produced by a chart determinant gives an exact split
of the input, independently of the heuristic used to choose the pair. -/
theorem chart_collision_gcd_certificate (N r : ℕ)
    (hlo : 1 < Nat.gcd r N) (hhi : Nat.gcd r N < N) :
    ∃ q, N = Nat.gcd r N * q ∧ 1 < Nat.gcd r N ∧ Nat.gcd r N < N := by
  exact FactoringResearch.projective_collision_gcd_certificate N r hlo hhi

end ProjectiveCollision
