import RequestProject.PythagoreanDescent

set_option autoImplicit false

namespace PythagoreanDescent

/-- The root of the Berggren tree. -/
def root : Triple := ⟨3, 4, 5⟩

/-- If the odd-oriented leg is fixed to `N`, requiring one parent step to land
at the root does not leave `N` free: each branch gives one particular child. -/
theorem one_step_to_root (N b c : ℤ) :
    (parent .U ⟨N, b, c⟩ = root ↔ N = 5 ∧ b = 12 ∧ c = 13) ∧
    (parent .A ⟨N, b, c⟩ = root ↔ N = 21 ∧ b = 20 ∧ c = 29) ∧
    (parent .D ⟨N, b, c⟩ = root ↔ N = 15 ∧ b = 8 ∧ c = 17) := by
  simp only [parent, root, parentU, parentA, parentD]
  constructor
  · constructor
    · intro h
      simp only [Triple.mk.injEq] at h
      omega
    · intro ⟨hN, hb, hc⟩
      simp only [Triple.mk.injEq]
      simp [hN, hb, hc]
  constructor
  · constructor
    · intro h
      simp only [Triple.mk.injEq] at h
      omega
    · intro ⟨hN, hb, hc⟩
      simp only [Triple.mk.injEq]
      simp [hN, hb, hc]
  · constructor
    · intro h
      simp only [Triple.mk.injEq] at h
      omega
    · intro ⟨hN, hb, hc⟩
      simp only [Triple.mk.injEq]
      simp [hN, hb, hc]

/-- Substituting the root after two inverse steps merely enumerates the nine
nodes at depth two.  In particular, their fixed odd legs are
`7,39,33,55,119,65,45,77,35`. -/
theorem depth_two_from_root :
    child .U (child .U root) = ⟨7, 24, 25⟩ ∧
    child .U (child .A root) = ⟨39, 80, 89⟩ ∧
    child .U (child .D root) = ⟨33, 56, 65⟩ ∧
    child .A (child .U root) = ⟨55, 48, 73⟩ ∧
    child .A (child .A root) = ⟨119, 120, 169⟩ ∧
    child .A (child .D root) = ⟨65, 72, 97⟩ ∧
    child .D (child .U root) = ⟨45, 28, 53⟩ ∧
    child .D (child .A root) = ⟨77, 36, 85⟩ ∧
    child .D (child .D root) = ⟨35, 12, 37⟩ := by
  norm_num [root, child, childU, childA, childD]

/-- Polynomial coordinates for the known thin-triple chain.  At integer index
`k`, its odd leg is `2k+3`. -/
def thinAt (k : ℤ) : Triple :=
  ⟨2 * k + 3, 2 * k ^ 2 + 6 * k + 4, 2 * k ^ 2 + 6 * k + 5⟩

/-- The root starts the thin chain. -/
theorem thinAt_zero : thinAt 0 = root := by
  norm_num [thinAt, root]

/-- Repeated `U` children advance along the thin chain. -/
theorem childU_thinAt (k : ℤ) : child .U (thinAt k) = thinAt (k + 1) := by
  simp [child, thinAt, childU]
  ring_nf
  simp

/-- Fixing only the leg `N` leaves the missing coordinates constrained by a
factorization of `N²`; this is the algebraic obstruction to beginning a
unique parent descent from `N` alone. -/
theorem fixed_leg_completion (N b c : ℤ)
    (h : N ^ 2 + b ^ 2 = c ^ 2) :
    (c - b) * (c + b) = N ^ 2 := by
  exact pythagorean_difference N b c h

end PythagoreanDescent
