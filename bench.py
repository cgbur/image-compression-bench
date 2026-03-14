#!/usr/bin/env python3
"""
Benchmark image compression across formats, qualities, and resolutions.

Usage:
    python bench.py <source_image> [source_image2 ...]
    python bench.py sources/*.avif -o results/
    python bench.py sources/*.avif --plot  # also generate plots

Measures global quality: upscale compressed image back to original resolution,
compare against original using SSIMULACRA 2. This captures both compression
artifacts and resolution loss together.
"""

import argparse
import csv
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

FORMATS = {
    "avif": {"ext": "avif", "qualities": [10, 20, 30, 40, 50, 60, 70, 80, 90, 95]},
    "jxl": {"ext": "jxl", "qualities": [10, 20, 30, 40, 50, 60, 70, 80, 90, 95]},
    "webp": {"ext": "webp", "qualities": [10, 20, 30, 40, 50, 60, 70, 80, 90, 95]},
    "jpeg": {"ext": "jpg", "qualities": [10, 20, 30, 40, 50, 60, 70, 80, 90, 95]},
}
WIDTHS = [600, 800, 1200, 1600, 2000, 2400, 3000, 3500, 4000]

SSIMULACRA2 = Path(__file__).parent / "ssimulacra2"

FIELDNAMES = [
    "source", "format", "quality", "width", "size_kb",
    "ssimulacra2",
]


def _measure_ssimulacra2(ref: Path, test_png: Path) -> float:
    """Run ssimulacra2, return score (-inf..100)."""
    result = subprocess.run(
        [str(SSIMULACRA2), str(ref), str(test_png)],
        capture_output=True, text=True,
    )
    for line in result.stdout.strip().splitlines():
        try:
            return float(line)
        except ValueError:
            continue
    raise RuntimeError(f"ssimulacra2 failed: {result.stderr}")


def make_reference(src: Path, width: int, tmpdir: Path) -> Path:
    ref = tmpdir / f"ref-{width}w.png"
    if not ref.exists():
        subprocess.run(
            ["magick", str(src), "-resize", f"{width}x", str(ref)],
            check=True, capture_output=True,
        )
    return ref


def get_original_dims(src: Path) -> tuple[int, int]:
    result = subprocess.run(
        ["magick", "identify", "-format", "%wx%h", str(src)],
        capture_output=True, text=True, check=True,
    )
    w, h = result.stdout.strip().split("x")
    return int(w), int(h)


def encode_and_measure(
    src: Path, fmt: str, quality: int, width: int,
    ref_width: int, ref_height: int, tmpdir: Path,
) -> dict:
    ext = FORMATS[fmt]["ext"]
    tag = f"{fmt}-q{quality}-{width}w"
    out = tmpdir / f"{tag}.{ext}"
    global_png = tmpdir / f"_global-{tag}.png"
    ref_global = tmpdir / f"ref-{ref_width}w.png"

    # Encode: resize + compress
    subprocess.run(
        ["magick", str(src), "-resize", f"{width}x", "-quality", str(quality), str(out)],
        check=True, capture_output=True,
    )

    size_kb = out.stat().st_size / 1024

    # Global comparison: upscale to reference dimensions
    subprocess.run(
        ["magick", str(out), "-resize", f"{ref_width}x{ref_height}!", str(global_png)],
        check=True, capture_output=True,
    )

    score = _measure_ssimulacra2(ref_global, global_png)

    # Cleanup
    for f in [out, global_png]:
        f.unlink(missing_ok=True)

    return {
        "source": src.name,
        "format": fmt,
        "quality": quality,
        "width": width,
        "size_kb": round(size_kb, 1),
        "ssimulacra2": round(score, 4),
    }


def run_benchmark(
    sources: list[Path], outdir: Path, max_workers: int = 4, ref_width: int | None = 2000,
) -> list[dict]:
    outdir.mkdir(parents=True, exist_ok=True)
    all_results = []

    for src in sources:
        print(f"\nBenchmarking: {src.name}")
        orig_width, orig_height = get_original_dims(src)
        print(f"  Original size: {orig_width}x{orig_height}")

        # Determine reference size for comparison
        if ref_width is None or ref_width >= orig_width:
            rw, rh = orig_width, orig_height
        else:
            rw = ref_width
            rh = round(orig_height * ref_width / orig_width)
        print(f"  Reference size: {rw}x{rh}")

        # Clip benchmark widths to not exceed reference width
        widths = [w for w in WIDTHS if w <= rw]
        if not widths:
            widths = [rw]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create references at all widths + reference width
            widths_needed = set(widths) | {rw}
            for w in widths_needed:
                make_reference(src, w, tmpdir)

            tasks = []
            for fmt, cfg in FORMATS.items():
                for q in cfg["qualities"]:
                    for w in widths:
                        tasks.append((src, fmt, q, w, rw, rh, tmpdir))

            print(f"  {len(tasks)} variants ({len(FORMATS)} formats x {len(widths)} widths)...")

            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(encode_and_measure, *t): t for t in tasks}
                done = 0
                for future in as_completed(futures):
                    done += 1
                    if done % 30 == 0 or done == len(tasks):
                        print(f"  {done}/{len(tasks)}")
                    try:
                        all_results.append(future.result())
                    except Exception as e:
                        t = futures[future]
                        print(f"  FAILED: {t[1]} q{t[2]} {t[3]}w: {e}")

    csv_path = outdir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(sorted(all_results, key=lambda r: r["size_kb"]))

    print(f"\nWrote {len(all_results)} results to {csv_path}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Benchmark image compression formats")
    parser.add_argument("sources", nargs="+", type=Path, help="Source image(s)")
    parser.add_argument("-o", "--outdir", type=Path, default=Path("results"),
                        help="Output directory (default: ./results/)")
    parser.add_argument("-j", "--jobs", type=int, default=None,
                        help="Max parallel workers (default: all CPUs)")
    parser.add_argument("--ref-width", type=int, default=2000,
                        help="Reference width for quality comparison (default: 2000, use 0 for original)")
    parser.add_argument("--plot", action="store_true",
                        help="Also generate plots after benchmarking")
    args = parser.parse_args()

    for s in args.sources:
        if not s.exists():
            print(f"Source not found: {s}", file=sys.stderr)
            sys.exit(1)

    rw = None if args.ref_width == 0 else args.ref_width
    results = run_benchmark(args.sources, args.outdir, max_workers=args.jobs, ref_width=rw)

    if args.plot:
        from plot import generate_all
        generate_all(results, args.outdir)


if __name__ == "__main__":
    main()
