"""CLI entry point: ``python -m gradtuity.bench [pattern]``.

Optionally pass a substring to filter specs by name. Outputs a markdown
table to stdout and writes ``bench_results.{md,csv}`` to the cwd.
"""

import sys
from pathlib import Path

from .runner import format_csv, format_markdown, run_spec
from .specs import all_specs


def main(argv: list[str]) -> int:
    pattern = argv[1] if len(argv) > 1 else ""
    specs = [s for s in all_specs() if pattern in s.name]
    if not specs:
        print(f"No specs match pattern {pattern!r}", file=sys.stderr)
        return 1

    print(f"Running {len(specs)} benchmark spec(s)...\n")
    results = []
    for spec in specs:
        print(f"  {spec.name} [{spec.shape_label}] ...", end=" ", flush=True)
        r = run_spec(spec)
        results.append(r)
        ratio = r.triton_ms / r.cuda_ms if r.cuda_ms > 0 else float("nan")
        winner = "CUDA" if ratio > 1.05 else ("Triton" if ratio < 0.95 else "tie")
        print(
            f"triton={r.triton_ms:.4f} ms, cuda={r.cuda_ms:.4f} ms "
            f"({ratio:.2f}× {winner}), max|Δ|={r.max_abs_diff:.2e}"
        )

    md = format_markdown(results)
    csv = format_csv(results)
    print()
    print(md)

    out_dir = Path("bench_results")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "summary.md").write_text(md + "\n")
    (out_dir / "summary.csv").write_text(csv + "\n")
    print(f"\nWrote {out_dir / 'summary.md'} and {out_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
