"""sign_test.py - paired comparison of two models (ties removed)."""
from math import comb


def sign_test_p(wins_a: int, wins_b: int) -> float:
    """Two-sided sign test p-value (ties already removed)."""
    n, k = wins_a + wins_b, min(wins_a, wins_b)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


if __name__ == "__main__":
    print(round(sign_test_p(14, 5), 3))   # 0.064
    print(round(sign_test_p(30, 10), 3))  # 0.002
    print(round(sign_test_p(22, 14), 3))  # exercise 36-8
