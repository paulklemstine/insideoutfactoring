import RequestProject.ThinAscension
import RequestProject.ResearchControls

set_option autoImplicit false

namespace AuxiliaryOrbit

open PythagoreanDescent

/-- Replay a public branch word from a supplied triple.  This is the exact
mathematical object sampled modulo each auxiliary prime in AQ-010. -/
def replay : List Branch → Triple → Triple
  | [], v => v
  | s :: ss, v => replay ss (child s v)

/-- Congruent starting points remain congruent after replaying the same public
branch word.  In particular, auxiliary-prime features depend only on the
public reduced orbit, not on a hidden factorization of the input. -/
theorem replay_preserves_modEq (p : ℤ) (word : List Branch) (v w : Triple)
    (ha : Int.ModEq p v.a w.a)
    (hb : Int.ModEq p v.b w.b)
    (hc : Int.ModEq p v.c w.c) :
    Int.ModEq p (replay word v).a (replay word w).a ∧
    Int.ModEq p (replay word v).b (replay word w).b ∧
    Int.ModEq p (replay word v).c (replay word w).c := by
  have hchild : ∀ (s : Branch) (v w : Triple),
      Int.ModEq p v.a w.a → Int.ModEq p v.b w.b → Int.ModEq p v.c w.c →
      Int.ModEq p (child s v).a (child s w).a ∧
      Int.ModEq p (child s v).b (child s w).b ∧
      Int.ModEq p (child s v).c (child s w).c := by
    intro s v w ha hb hc
    cases s with
    | U =>
      simp [child, childU]
      refine ⟨?_, ?_, ?_⟩
      · -- v.a - 2 * v.b + 2 * v.c = (v.a - 2 * v.b) + 2 * v.c
        exact Int.ModEq.add (Int.ModEq.sub ha (Int.ModEq.mul_left 2 hb)) (Int.ModEq.mul_left 2 hc)
      · -- 2 * v.a - v.b + 2 * v.c = (2 * v.a - v.b) + 2 * v.c
        exact Int.ModEq.add (Int.ModEq.sub (Int.ModEq.mul_left 2 ha) hb) (Int.ModEq.mul_left 2 hc)
      · -- 2 * v.a - 2 * v.b + 3 * v.c = (2 * v.a - 2 * v.b) + 3 * v.c
        exact Int.ModEq.add (Int.ModEq.sub (Int.ModEq.mul_left 2 ha) (Int.ModEq.mul_left 2 hb)) (Int.ModEq.mul_left 3 hc)
    | A =>
      simp [child, childA]
      refine ⟨?_, ?_, ?_⟩
      · -- v.a + 2 * v.b + 2 * v.c = (v.a + 2 * v.b) + 2 * v.c
        exact Int.ModEq.add (Int.ModEq.add ha (Int.ModEq.mul_left 2 hb)) (Int.ModEq.mul_left 2 hc)
      · -- 2 * v.a + v.b + 2 * v.c = (2 * v.a + v.b) + 2 * v.c
        exact Int.ModEq.add (Int.ModEq.add (Int.ModEq.mul_left 2 ha) hb) (Int.ModEq.mul_left 2 hc)
      · -- 2 * v.a + 2 * v.b + 3 * v.c = (2 * v.a + 2 * v.b) + 3 * v.c
        exact Int.ModEq.add (Int.ModEq.add (Int.ModEq.mul_left 2 ha) (Int.ModEq.mul_left 2 hb)) (Int.ModEq.mul_left 3 hc)
    | D =>
      simp [child, childD]
      refine ⟨?_, ?_, ?_⟩
      · -- -v.a + 2 * v.b + 2 * v.c = (-v.a + 2 * v.b) + 2 * v.c
        exact Int.ModEq.add (Int.ModEq.add (Int.ModEq.neg ha) (Int.ModEq.mul_left 2 hb)) (Int.ModEq.mul_left 2 hc)
      · -- -(2 * v.a) + v.b + 2 * v.c = (-(2 * v.a) + v.b) + 2 * v.c
        exact Int.ModEq.add (Int.ModEq.add (Int.ModEq.neg (Int.ModEq.mul_left 2 ha)) hb) (Int.ModEq.mul_left 2 hc)
      · -- -(2 * v.a) + 2 * v.b + 3 * v.c = (-(2 * v.a) + 2 * v.b) + 3 * v.c
        exact Int.ModEq.add (Int.ModEq.add (Int.ModEq.neg (Int.ModEq.mul_left 2 ha)) (Int.ModEq.mul_left 2 hb)) (Int.ModEq.mul_left 3 hc)
  induction word generalizing v w with
  | nil => exact ⟨ha, hb, hc⟩
  | cons s ss ih =>
    simp only [replay]
    exact ih (child s v) (child s w) (hchild s v w ha hb hc).1 (hchild s v w ha hb hc).2.1 (hchild s v w ha hb hc).2.2

/-- A successful experimental outcome is accepted only at the exact GCD
boundary, independently of any auxiliary-prime score used to schedule it. -/
theorem scheduled_residue_exact_split (N r : ℕ)
    (hlo : 1 < Nat.gcd r N) (hhi : Nat.gcd r N < N) :
    ∃ q, N = Nat.gcd r N * q ∧ 1 < Nat.gcd r N ∧ Nat.gcd r N < N := by
  exact ⟨N / Nat.gcd r N, (Nat.mul_div_cancel' (Nat.gcd_dvd_right r N)).symm, hlo, hhi⟩

end AuxiliaryOrbit
