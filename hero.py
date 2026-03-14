#!/usr/bin/env python3
"""
Generate the hero/icon image: a grid showing all four formats across quality levels.

Columns = formats (JPEG, WebP, JPEG XL, AVIF), rows = quality levels (low at bottom,
high at top). Each cell is cropped from the encoded image at its grid position so the
whole thing reads as one continuous scene.

Usage:
    python hero.py sources/film_castle.avif -o hero.png
    python hero.py sources/film_castle.avif -o hero.avif --width 1600
"""

import argparse
import subprocess
import tempfile
from pathlib import Path


FORMATS = [
    ("JPEG", "jpg"),
    ("JPEG XL", "jxl"),
    ("AVIF", "avif"),
    ("WebP", "webp"),
]

# Log-biased quality levels: more samples at the low end where differences are visible
QUALITIES = [1, 5, 10, 15, 30, 50, 75, 95]


def get_dims(path: Path) -> tuple[int, int]:
    result = subprocess.run(
        ["magick", "identify", "-format", "%wx%h", str(path)],
        capture_output=True, text=True, check=True,
    )
    w, h = result.stdout.strip().split("x")
    return int(w), int(h)


def generate_hero(src: Path, out: Path, width: int = 1600):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create reference at target width
        ref = tmpdir / "ref.png"
        subprocess.run(
            ["magick", str(src), "-resize", f"{width}x", str(ref)],
            check=True, capture_output=True,
        )
        w, h = get_dims(ref)

        num_f = len(FORMATS)
        num_q = len(QUALITIES)
        col_w = w // num_f
        row_h = h // num_q

        # Encode at each format/quality and crop the right cell
        col_paths = []
        for fi, (label, ext) in enumerate(FORMATS):
            x = fi * col_w
            cell_paths = []
            for qi, q in enumerate(QUALITIES):
                # Row 0 = highest quality (top), so invert
                y = (num_q - 1 - qi) * row_h

                encoded = tmpdir / f"{ext}_q{q}.{ext}"
                subprocess.run(
                    ["magick", str(ref), "-quality", str(q), str(encoded)],
                    check=True, capture_output=True,
                )

                cell = tmpdir / f"cell_{ext}_{qi}.png"
                subprocess.run(
                    ["magick", str(encoded),
                     "-crop", f"{col_w}x{row_h}+{x}+{y}", "+repage",
                     str(cell)],
                    check=True, capture_output=True,
                )
                cell_paths.append(cell)

            # Stack cells top to bottom (highest quality first)
            col_img = tmpdir / f"col_{ext}.png"
            subprocess.run(
                ["magick", *[str(c) for c in reversed(cell_paths)],
                 "-append", str(col_img)],
                check=True, capture_output=True,
            )
            col_paths.append(col_img)

        # Join columns left to right
        joined = tmpdir / "joined.png"
        subprocess.run(
            ["magick", *[str(c) for c in col_paths], "+append", str(joined)],
            check=True, capture_output=True,
        )

        # Add dividers and labels
        draw_args = []

        # Vertical dividers between formats
        for i in range(1, num_f):
            x = i * col_w
            draw_args += ["-draw", f"rectangle {x-1},0 {x},{h}"]

        # Horizontal dividers between quality rows
        for i in range(1, num_q):
            y = i * row_h
            draw_args += ["-draw", f"rectangle 0,{y-1} {w},{y}"]

        # Format labels at top
        label_args = [
            "-fill", "white", "-stroke", "rgba(0,0,0,0.5)",
            "-strokewidth", "1.2", "-pointsize", str(max(20, width // 60)),
            "-gravity", "NorthWest",
        ]
        for fi, (label, _) in enumerate(FORMATS):
            cx = fi * col_w + col_w // 2 - len(label) * max(5, width // 120)
            label_args += ["-annotate", f"+{cx}+8", label]

        # Quality labels on right edge, centered in each row
        q_label_args = ["-pointsize", str(max(16, width // 80))]
        for qi, q in enumerate(QUALITIES):
            y = (num_q - 1 - qi) * row_h + row_h // 2 - 10
            q_label_args += ["-annotate", f"+{w - max(40, width // 30)}+{y}", f"q{q}"]

        subprocess.run(
            ["magick", str(joined),
             "-fill", "rgba(0,0,0,0.2)", *draw_args,
             *label_args, *q_label_args,
             "-depth", "8",
             str(out)],
            check=True, capture_output=True,
        )

    print(f"Wrote {out} ({out.stat().st_size / 1024:.0f}KB)")


def main():
    parser = argparse.ArgumentParser(description="Generate hero image grid")
    parser.add_argument("source", type=Path, help="Source image")
    parser.add_argument("-o", "--output", type=Path, default=Path("hero.png"),
                        help="Output path (default: hero.png)")
    parser.add_argument("--width", type=int, default=1600,
                        help="Output width in pixels (default: 1600)")
    args = parser.parse_args()

    generate_hero(args.source, args.output, width=args.width)


if __name__ == "__main__":
    main()
