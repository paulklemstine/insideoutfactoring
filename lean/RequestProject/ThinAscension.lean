import RequestProject.LegNDescent

set_option autoImplicit false

namespace PythagoreanDescent

/-- The scale-free residue seed underlying the thin triple modulo an odd `N`. -/
def normalizedThinSeed : Triple := ⟨0, -1, 1⟩

/-- Twice the thin triple, written without division. -/
def doubledThin (N : ℤ) : Triple := ⟨2 * N, N ^ 2 - 1, N ^ 2 + 1⟩

/-- The doubled thin triple reduces to the fixed normalized seed modulo `N`. -/
theorem doubledThin_mod_normalized (N : ℤ) :
    Int.ModEq N (doubledThin N).a normalizedThinSeed.a ∧
    Int.ModEq N (doubledThin N).b normalizedThinSeed.b ∧
    Int.ModEq N (doubledThin N).c normalizedThinSeed.c := by
  simp [doubledThin, normalizedThinSeed, Int.modEq_iff_dvd]
  ring_nf
  exact dvd_neg.mpr (dvd_pow_self N two_ne_zero)

/-- Integer-linear Berggren ascension preserves coordinatewise congruence. -/
theorem child_preserves_modEq (N : ℤ) (s : Branch) (v w : Triple)
    (ha : Int.ModEq N v.a w.a)
    (hb : Int.ModEq N v.b w.b)
    (hc : Int.ModEq N v.c w.c) :
    Int.ModEq N (child s v).a (child s w).a ∧
    Int.ModEq N (child s v).b (child s w).b ∧
    Int.ModEq N (child s v).c (child s w).c := by
  refine ⟨?_, ?_, ?_⟩
  all_goals cases s <;> simp [child, childU, childA, childD]
  all_goals
    first
    | exact Int.ModEq.add (Int.ModEq.sub ha (Int.ModEq.mul_left _ hb)) (Int.ModEq.mul_left _ hc)
    | exact Int.ModEq.add (Int.ModEq.sub (Int.ModEq.mul_left _ ha) hb) (Int.ModEq.mul_left _ hc)
    | exact Int.ModEq.add (Int.ModEq.sub (Int.ModEq.mul_left _ ha) (Int.ModEq.mul_left _ hb)) (Int.ModEq.mul_left _ hc)
    | exact Int.ModEq.add (Int.ModEq.add ha (Int.ModEq.mul_left _ hb)) (Int.ModEq.mul_left _ hc)
    | exact Int.ModEq.add (Int.ModEq.add (Int.ModEq.mul_left _ ha) hb) (Int.ModEq.mul_left _ hc)
    | exact Int.ModEq.add (Int.ModEq.add (Int.ModEq.mul_left _ ha) (Int.ModEq.mul_left _ hb)) (Int.ModEq.mul_left _ hc)
    | exact Int.ModEq.add (Int.ModEq.add (Int.ModEq.neg ha) (Int.ModEq.mul_left _ hb)) (Int.ModEq.mul_left _ hc)
    | exact Int.ModEq.add (Int.ModEq.add (Int.ModEq.neg (Int.ModEq.mul_left _ ha)) hb) (Int.ModEq.mul_left _ hc)
    | exact Int.ModEq.add (Int.ModEq.add (Int.ModEq.neg (Int.ModEq.mul_left _ ha)) (Int.ModEq.mul_left _ hb)) (Int.ModEq.mul_left _ hc)

/-- The first normalized `U` child gives the familiar `(4,3,5)` coefficients,
while the `A` and `D` children coincide at `(0,1,1)`. -/
theorem normalizedThinSeed_children :
    child .U normalizedThinSeed = ⟨4, 3, 5⟩ ∧
    child .A normalizedThinSeed = ⟨0, 1, 1⟩ ∧
    child .D normalizedThinSeed = ⟨0, 1, 1⟩ := by
  exact ⟨rfl, rfl, rfl⟩

end PythagoreanDescent
