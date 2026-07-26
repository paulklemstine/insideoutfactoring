import RequestProject.AdaptivePortfolio
import RequestProject.InsideOut

set_option autoImplicit false

namespace AdaptivePortfolio

/-- All MOSAIC workers share one candidate stream.  Concatenating streams from
arithmetic, sequence, geometric, or spectral schedulers cannot weaken the exact
certificate attached to the first accepted proposal. -/
theorem combined_stream_sound {N d : ℕ} {streams : List (List ℕ)}
    (h : firstValid N streams.flatten = some d) :
    1 < d ∧ d < N ∧ d ∣ N := by
  exact firstValid_sound h

/-- A successful combined stream gives an exact factor/cofactor decomposition,
independently of which worker generated the winning candidate. -/
theorem combined_stream_yields_split {N d : ℕ} {streams : List (List ℕ)}
    (h : firstValid N streams.flatten = some d) :
    ∃ q, N = d * q ∧ 1 < d ∧ d < N := by
  exact firstValid_yields_split h

/-- Scheduling an entirely rejected experimental epoch before the established
fallback leaves the fallback's selected result unchanged. -/
theorem rejected_epoch_fallback_invariant {N : ℕ}
    {experimental fallback : List ℕ}
    (hreject : ∀ d ∈ experimental, validFactor N d = false) :
    firstValid N (experimental ++ fallback) = firstValid N fallback := by
  exact rejected_prefix_invariant hreject

/-- Any nontrivial GCD residue emitted onto the common residue bus is itself a
certified proper divisor. -/
theorem residue_bus_gcd_sound (N x : ℕ)
    (hlo : 1 < Nat.gcd x N) (hhi : Nat.gcd x N < N) :
    1 < Nat.gcd x N ∧ Nat.gcd x N < N ∧ Nat.gcd x N ∣ N := by
  exact ⟨hlo, hhi, Nat.gcd_dvd_right x N⟩

/-- The residue-bus certificate also supplies the complementary cofactor. -/
theorem residue_bus_gcd_split (N x : ℕ)
    (hlo : 1 < Nat.gcd x N) (hhi : Nat.gcd x N < N) :
    ∃ q, N = Nat.gcd x N * q ∧ 1 < Nat.gcd x N ∧ Nat.gcd x N < N := by
  refine ⟨N / Nat.gcd x N, ?_, hlo, hhi⟩
  exact (Nat.mul_div_cancel' (Nat.gcd_dvd_right x N)).symm

end AdaptivePortfolio
