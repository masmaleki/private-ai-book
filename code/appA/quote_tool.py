"""Appendix A: a small command-line script with argparse (example data)."""
import argparse
from pathlib import Path


def margin_price(cost: float, margin_pct: float) -> float:
    """Selling price for a target margin on the selling price."""
    if not 0 <= margin_pct < 100:
        raise ValueError("margin_pct must be in [0, 100)")
    return cost / (1 - margin_pct / 100)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Suggest a unit price from cost and margin")
    ap.add_argument("cost", type=float, help="unit cost in EUR")
    ap.add_argument("--margin", type=float, default=12.0,
                    help="target margin in percent")
    ap.add_argument("--out", type=Path, help="optional file to append to")
    args = ap.parse_args()
    price = margin_price(args.cost, args.margin)
    line = (f"cost={args.cost:.4f} margin={args.margin:.1f}% "
            f"price={price:.4f} EUR")
    print(line)
    if args.out:
        with args.out.open("a", encoding="utf-8") as f:  # explicit encoding
            f.write(line + "\n")


if __name__ == "__main__":
    main()
