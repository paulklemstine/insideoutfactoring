import RequestProject.StructuredRelations

set_option autoImplicit false

namespace LargePrimeCompletion

/-- The number `E - V + C` used by LP-013 is nonnegative whenever a finite
multigraph has at least `V - C` edges.  The graph analysis establishes this
premise component by component. -/
theorem cycleRank_nonnegative (edges vertices components : ℕ)
    (hforest : vertices ≤ edges + components) :
    vertices - components ≤ edges := by
  omega

/-- Adding one edge whose endpoints were already connected increments the
cycle-space count by exactly one (with fixed vertex/component counts). -/
theorem cycleRank_add_internal_edge (edges vertices components : ℕ)
    (hforest : vertices ≤ edges + components) :
    (edges + 1 + components) - vertices =
      (edges + components - vertices) + 1 := by
  omega

/-- Adding a bridge between two components leaves the cycle-space count
unchanged: the edge count rises once and the component count falls once. -/
theorem cycleRank_add_bridge (edges vertices components : ℕ)
    (hcomponents : 0 < components) :
    (edges + 1 + (components - 1)) - vertices =
      (edges + components) - vertices := by
  have hcount : edges + 1 + (components - 1) = edges + components := by
    omega
  rw [hcount]

/-- Two relations carrying the same large-prime cofactor cancel that cofactor
in their product, leaving the product of their factor-base-smooth parts times a
square.  This is the algebraic kernel of one-large-prime completion. -/
theorem repeated_large_prime_completion (smooth₁ smooth₂ p : ℕ) :
    (smooth₁ * p) * (smooth₂ * p) = (smooth₁ * smooth₂) * p ^ 2 := by
  ring

/-- Around any closed residual-prime walk, every vertex prime occurs twice in
the product.  This four-edge instance records the two-large-prime cycle kernel
used by the graph analysis. -/
theorem four_cycle_completion (s₁ s₂ s₃ s₄ p q r t : ℕ) :
    (s₁ * (p * q)) * (s₂ * (q * r)) * (s₃ * (r * t)) * (s₄ * (t * p)) =
      (s₁ * s₂ * s₃ * s₄) * (p * q * r * t) ^ 2 := by
  ring

/-- A relation retained by the one/two-large-prime variant consists of a
factor-base part, two residual vertices (`1,p` represents a one-large-prime
edge), and a square root modulo the integer being factored. -/
structure PartialRelation where
  smoothPart : ℕ
  leftPrime : ℕ
  rightPrime : ℕ
  root : ℕ

/-- Exact arithmetic meaning of a partial relation. -/
def PartialRelation.Valid (N : ℕ) (R : PartialRelation) : Prop :=
  Nat.ModEq N (R.smoothPart * (R.leftPrime * R.rightPrime)) (R.root ^ 2)

/-- Two one-large-prime relations with the same residual prime combine to a
square congruence as soon as their smooth parts combine to a square. -/
theorem one_large_prime_square_congruence (N s₁ s₂ p x₁ x₂ y : ℕ)
    (h₁ : Nat.ModEq N (s₁ * p) (x₁ ^ 2))
    (h₂ : Nat.ModEq N (s₂ * p) (x₂ ^ 2))
    (hsmooth : s₁ * s₂ = y ^ 2) :
    Nat.ModEq N ((y * p) ^ 2) ((x₁ * x₂) ^ 2) := by
  have hmul := Nat.ModEq.mul h₁ h₂
  simp only [sq] at *
  ring_nf at hmul ⊢
  rw [show s₁ * p ^ 2 * s₂ = p ^ 2 * (s₁ * s₂) by ring] at hmul
  rw [hsmooth] at hmul
  convert hmul using 2 <;> ring

/-- A four-edge two-large-prime cycle likewise combines to a square
congruence. This is the formal counterpart of the graph-cycle completion
counted in LP-013. -/
theorem four_cycle_square_congruence
    (N s₁ s₂ s₃ s₄ p q r t x₁ x₂ x₃ x₄ y : ℕ)
    (h₁ : Nat.ModEq N (s₁ * (p * q)) (x₁ ^ 2))
    (h₂ : Nat.ModEq N (s₂ * (q * r)) (x₂ ^ 2))
    (h₃ : Nat.ModEq N (s₃ * (r * t)) (x₃ ^ 2))
    (h₄ : Nat.ModEq N (s₄ * (t * p)) (x₄ ^ 2))
    (hsmooth : s₁ * s₂ * s₃ * s₄ = y ^ 2) :
    Nat.ModEq N ((y * (p * q * r * t)) ^ 2)
      ((x₁ * x₂ * x₃ * x₄) ^ 2) := by
  have hmul := Nat.ModEq.mul (Nat.ModEq.mul (Nat.ModEq.mul h₁ h₂) h₃) h₄
  have hcycle :
      s₁ * (p * q) * (s₂ * (q * r)) * (s₃ * (r * t)) * (s₄ * (t * p)) =
        (s₁ * s₂ * s₃ * s₄) * (p * q * r * t) ^ 2 :=
    four_cycle_completion s₁ s₂ s₃ s₄ p q r t
  rw [hcycle] at hmul
  rw [hsmooth] at hmul
  have lhs_eq : y ^ 2 * (p * q * r * t) ^ 2 = (y * (p * q * r * t)) ^ 2 := by ring
  have rhs_eq : x₁ ^ 2 * x₂ ^ 2 * x₃ ^ 2 * x₄ ^ 2 = (x₁ * x₂ * x₃ * x₄) ^ 2 := by ring
  rw [←lhs_eq, ←rhs_eq]
  exact hmul

/-- Completion is useful for factoring only through the standard exact GCD
certificate: a proper GCD from the completed square congruence supplies an
exact factor/cofactor split. -/
theorem completed_relation_exact_split (N x y : ℕ)
    (hcong : Nat.ModEq N (x ^ 2) (y ^ 2))
    (hlo : 1 < Nat.gcd (x + y) N)
    (hhi : Nat.gcd (x + y) N < N) :
    Nat.ModEq N (x ^ 2) (y ^ 2) ∧
      ∃ q, N = Nat.gcd (x + y) N * q ∧
        1 < Nat.gcd (x + y) N ∧ Nat.gcd (x + y) N < N := by
  refine ⟨hcong, N / Nat.gcd (x + y) N, ?_, hlo, hhi⟩
  rw [Nat.mul_div_cancel' (Nat.gcd_dvd_right _ _)]

end LargePrimeCompletion
