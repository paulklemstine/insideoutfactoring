import Mathlib

set_option autoImplicit false

namespace AdaptivePortfolio

/-- The exact gate shared by every heuristic worker. A method name, score, or
physical interpretation cannot bypass this predicate. -/
def validFactor (N d : ℕ) : Bool := 1 < d && d < N && N % d == 0

/-- Propositional meaning of the executable certificate gate. -/
theorem validFactor_eq_true_iff (N d : ℕ) :
    validFactor N d = true ↔ 1 < d ∧ d < N ∧ d ∣ N := by
  simp only [validFactor, Bool.and_eq_true, decide_eq_true_eq,
    beq_iff_eq, Nat.dvd_iff_mod_eq_zero]
  tauto

/-- Select the first certified proposal. This models the correctness boundary
of a portfolio independently of how workers produce or rank candidates. -/
def firstValid (N : ℕ) : List ℕ → Option ℕ
  | [] => none
  | d :: ds => if validFactor N d then some d else firstValid N ds

/-- Any proposal returned by the portfolio is a genuine proper divisor. -/
theorem firstValid_sound {N d : ℕ} {proposals : List ℕ}
    (h : firstValid N proposals = some d) :
    1 < d ∧ d < N ∧ d ∣ N := by
  induction proposals with
  | nil => simp [firstValid] at h
  | cons a as ih =>
      simp only [firstValid] at h
      split at h <;> rename_i hg
      · simp only [Option.some.injEq] at h
        subst a
        exact (validFactor_eq_true_iff N d).mp hg
      · exact ih h

/-- A certified split includes the complementary cofactor and exact product. -/
theorem firstValid_yields_split {N d : ℕ} {proposals : List ℕ}
    (h : firstValid N proposals = some d) :
    ∃ q, N = d * q ∧ 1 < d ∧ d < N := by
  have hs := firstValid_sound h
  exact ⟨N / d, (Nat.mul_div_cancel' hs.2.2).symm, hs.1, hs.2.1⟩

/-- Adding speculative candidates cannot alter a previously accepted answer
when they are all rejected by the exact gate. -/
theorem rejected_prefix_invariant {N : ℕ} {pre rest : List ℕ}
    (hbad : ∀ d ∈ pre, validFactor N d = false) :
    firstValid N (pre ++ rest) = firstValid N rest := by
  induction pre with
  | nil => rfl
  | cons a as ih =>
      simp only [List.mem_cons, forall_eq_or_imp] at hbad
      simp [firstValid, hbad.1, ih hbad.2]

end AdaptivePortfolio
