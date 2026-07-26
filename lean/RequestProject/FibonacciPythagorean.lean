import RequestProject.AdaptivePortfolio
import Mathlib.Data.Nat.Fib.Basic

set_option autoImplicit false

namespace FibonacciPythagorean

/-- The Lucas companion, represented over the integers to avoid truncated subtraction. -/
def lucas (n : ℕ) : ℤ := 2 * (Nat.fib (n + 1) : ℤ) - Nat.fib n

/-- Fibonacci-generated Pythagorean coordinates. -/
def fibA (n : ℕ) : ℕ := 2 * Nat.fib (n + 1) * Nat.fib (n + 2)
def fibB (n : ℕ) : ℕ := Nat.fib n * Nat.fib (n + 3)
def fibC (n : ℕ) : ℕ := Nat.fib (n + 1) ^ 2 + Nat.fib (n + 2) ^ 2

/-- Four consecutive Fibonacci values give a Pythagorean triple. -/
theorem fibonacci_pythagorean (n : ℕ) :
    fibA n ^ 2 + fibB n ^ 2 = fibC n ^ 2 := by
  simp [fibA, fibB, fibC]
  have h1 : Nat.fib (n + 2) = Nat.fib n + Nat.fib (n + 1) := by rw [Nat.fib_add_two]
  have h2 : Nat.fib (n + 3) = Nat.fib (n + 2) + Nat.fib (n + 1) := by
    have := @Nat.fib_add_two (n + 1)
    simp_all
    omega
  rw [h1, h2]
  ring_nf
  rw [show (2 + n : ℕ) = n + 2 by ring, h1]
  ring

/-- Its hypotenuse is the odd-index Fibonacci number stated in the paper. -/
theorem fibC_eq (n : ℕ) : fibC n = Nat.fib (2 * n + 3) := by
  simp [fibC]
  rw [show 2 * n + 3 = 2 * (n + 1) + 1 by ring]
  exact Nat.fib_two_mul_add_one (n + 1) ▸ by ring_nf

/-- The strong-divisibility law, exposed in the orientation useful to the search. -/
theorem fibonacci_strong_divisibility (m n : ℕ) :
    (Nat.fib m).gcd (Nat.fib n) = Nat.fib (m.gcd n) := by
  exact (Nat.fib_gcd m n).symm

/-- A rank-aligned Fibonacci residue gives a certified factor through one gcd,
provided the whole modulus does not divide the residue. -/
theorem fibonacci_gcd_certificate (N p k : ℕ)
    (hN : 0 < N) (hpN : p ∣ N) (hpk : p ∣ Nat.fib k) (hp : 1 < p)
    (hnot : ¬ N ∣ Nat.fib k) :
    1 < (Nat.fib k).gcd N ∧ (Nat.fib k).gcd N < N ∧ (Nat.fib k).gcd N ∣ N := by
  have hpd : p ∣ (Nat.fib k).gcd N := Nat.dvd_gcd hpk hpN
  have hgpos : 0 < (Nat.fib k).gcd N := Nat.gcd_pos_of_pos_right _ hN
  have hg_gt : 1 < (Nat.fib k).gcd N :=
    lt_of_lt_of_le hp (Nat.le_of_dvd hgpos hpd)
  have hg_dvd : (Nat.fib k).gcd N ∣ N := Nat.gcd_dvd_right _ _
  refine ⟨hg_gt, ?_, hg_dvd⟩
  have hle := Nat.gcd_le_right (Nat.fib k) hN
  apply lt_of_le_of_ne hle
  intro heq
  apply hnot
  rw [← Nat.gcd_eq_right_iff_dvd]
  exact heq

/-- Divisibility at an apparition index propagates to every multiple index. -/
theorem apparition_multiple {p z k : ℕ} (hz : p ∣ Nat.fib z) (hzk : z ∣ k) :
    p ∣ Nat.fib k := by
  exact dvd_trans hz (Nat.fib_dvd z k hzk)

/-- The first bridge coordinate batches the two adjacent Fibonacci probes. -/
theorem dvd_fibA_left {p n : ℕ} (h : p ∣ Nat.fib (n + 1)) : p ∣ fibA n := by
  simp [fibA]
  exact h.mul_left 2 |>.mul_right _

/-- The first bridge coordinate also batches the next Fibonacci probe. -/
theorem dvd_fibA_right {p n : ℕ} (h : p ∣ Nat.fib (n + 2)) : p ∣ fibA n := by
  simp only [fibA]
  exact h.trans (dvd_mul_left _ _)

end FibonacciPythagorean
