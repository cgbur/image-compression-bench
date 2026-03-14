#!/usr/bin/env python3
"""
Generate plots from benchmark results CSV.

Usage:
    python plot.py results/results.csv
    python plot.py results/results.csv -o results/
"""

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy import interpolate

FORMAT_COLORS = {
    "avif": "#e63946",
    "jxl": "#2a9d8f",
    "webp": "#457b9d",
    "jpeg": "#9a8c98",
}
FORMAT_LABELS = {
    "avif": "AVIF",
    "jxl": "JPEG XL",
    "webp": "WebP",
    "jpeg": "JPEG",
}
FORMAT_ORDER = ["avif", "jxl", "webp", "jpeg"]

# Module-level source directory for thumbnails, set by main() or generate_all()
SOURCE_DIR: Path | None = None

METRIC_LABELS = {
    "ssimulacra2": "SSIMULACRA 2",
}

# Module-level metric key, set by main() or generate_all()
METRIC = "ssimulacra2"

# Quality band thresholds for reference lines
QUALITY_BANDS = [(70, "high quality"), (80, "very high"), (90, "visually lossless")]


# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------

def init_style():
    """Set global matplotlib style. Call once before generating plots."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#cccccc",
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "axes.grid.which": "major",
        "grid.color": "#e8e8e8",
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlepad": 10,
        "axes.labelsize": 9,
        "axes.labelpad": 6,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#dddddd",
        "figure.titlesize": 13,
        "figure.titleweight": "bold",
        "lines.linewidth": 1.3,
        "lines.markersize": 4,
        "font.family": "sans-serif",
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def add_quality_bands(ax, side: str = "left"):
    """Add ssimulacra2 quality band reference lines to an axis."""
    for threshold, lbl in QUALITY_BANDS:
        ax.axhline(y=threshold, color="#e0e0e0", linestyle="--", linewidth=0.7)
        x_pos = 0.01 if side == "left" else 0.99
        ha = "left" if side == "left" else "right"
        ax.annotate(lbl, xy=(x_pos, threshold), xycoords=("axes fraction", "data"),
                    fontsize=6.5, color="#aaaaaa", va="bottom", ha=ha,
                    xytext=(4 if side == "left" else -4, 2), textcoords="offset points")


def save_fig(fig, outdir: Path, name: str):
    """Save figure with consistent settings."""
    fig.tight_layout()
    fig.savefig(outdir / name, bbox_inches="tight")
    plt.close()
    print(f"Wrote {outdir / name}")


def _render_thumbnail(ax, src_name: str):
    """Render a source image thumbnail (or fallback label) into an axes."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    label = src_name.rsplit(".", 1)[0].replace("_", " ")

    if SOURCE_DIR is not None:
        src_path = SOURCE_DIR / src_name
        if src_path.exists():
            try:
                with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
                    subprocess.run(
                        ["magick", str(src_path), "-resize", "300x300", tmp.name],
                        capture_output=True, check=True,
                    )
                    img = plt.imread(tmp.name)
                ax.imshow(img)
                ax.set_xlabel(label, fontsize=7, labelpad=2)
                return
            except Exception:
                pass

    ax.text(0.5, 0.5, label, ha="center", va="center", fontsize=8,
            transform=ax.transAxes, style="italic")


def _subplots_with_thumbnails(sources, ncols, col_width=3.5, row_height=4.5):
    """Create figure + axes grid, optionally prepending a thumbnail column."""
    nrows = len(sources)
    if SOURCE_DIR is not None:
        fig = plt.figure(figsize=(col_width * ncols + 2, row_height * nrows))
        gs = GridSpec(nrows, ncols + 1, figure=fig,
                      width_ratios=[0.25] + [1] * ncols)
        axes = np.empty((nrows, ncols), dtype=object)
        for r in range(nrows):
            ax_thumb = fig.add_subplot(gs[r, 0])
            _render_thumbnail(ax_thumb, sources[r][0])
            for c in range(ncols):
                axes[r, c] = fig.add_subplot(gs[r, c + 1])
        return fig, axes
    else:
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(col_width * ncols, row_height * nrows),
                                 squeeze=False)
        return fig, axes


def load_results(csv_path: Path) -> list[dict]:
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["quality"] = int(r["quality"])
        r["width"] = int(r["width"])
        r["size_kb"] = float(r["size_kb"])
        r["ssimulacra2"] = float(r["ssimulacra2"])
    return rows


def m(r: dict) -> float:
    """Get the active metric value from a result row."""
    return r[METRIC]


def metric_label() -> str:
    return METRIC_LABELS.get(METRIC, METRIC)


def formats_in(results: list[dict]) -> list[str]:
    present = set(r["format"] for r in results)
    return [f for f in FORMAT_ORDER if f in present]


def split_by_source(results: list[dict]) -> list[tuple[str, list[dict]]]:
    """Split results into (source_name, rows) pairs, preserving order of first appearance."""
    seen = {}
    for r in results:
        s = r["source"]
        if s not in seen:
            seen[s] = []
        seen[s].append(r)
    return list(seen.items())


def _pick_show_widths(results: list[dict], n: int = 4) -> list[int]:
    all_widths = sorted(set(r["width"] for r in results))
    if len(all_widths) <= n:
        return all_widths
    idxs = np.linspace(0, len(all_widths) - 1, n, dtype=int)
    return [all_widths[i] for i in idxs]


# ---------------------------------------------------------------------------
# Pareto: size vs SSIM
# ---------------------------------------------------------------------------

def plot_pareto(sources: list[tuple[str, list[dict]]], outdir: Path):
    nrows = len(sources)
    fig, axes = _subplots_with_thumbnails(sources, 1, col_width=11, row_height=6)

    for row, (src_name, results) in enumerate(sources):
        ax = axes[row][0]
        show_widths = _pick_show_widths(results)

        for fmt in formats_in(results):
            for i, w in enumerate(show_widths):
                pts = sorted(
                    [r for r in results if r["format"] == fmt and r["width"] == w],
                    key=lambda r: r["quality"],
                )
                if not pts:
                    continue
                label = FORMAT_LABELS[fmt] if i == 0 else None
                lw = 0.8 + 1.4 * (i / max(len(show_widths) - 1, 1))
                ax.plot(
                    [r["size_kb"] for r in pts], [m(r) for r in pts],
                    color=FORMAT_COLORS[fmt], alpha=0.75, linewidth=lw,
                    marker="o", markersize=3 + i, label=label,
                )
                best = pts[-1]
                ax.annotate(
                    f"{w}px", (best["size_kb"], m(best)),
                    fontsize=7, color=FORMAT_COLORS[fmt], alpha=0.8,
                    xytext=(6, 2), textcoords="offset points",
                )

        add_quality_bands(ax)
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_locator(ticker.FixedLocator([8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"{x/1024:.0f}MB" if x >= 1024 else f"{x:.0f}KB"
        ))
        ax.set_xlabel("File Size (KB)")
        ax.set_ylabel(metric_label())
        ax.set_title(src_name)
        ax.legend(loc="lower right")

    fig.suptitle(f"Image Compression: Size vs {metric_label()}")
    save_fig(fig, outdir, "pareto.png")


# ---------------------------------------------------------------------------
# By-width: faceted by resolution, rows = source images
# ---------------------------------------------------------------------------

def plot_by_width(sources: list[tuple[str, list[dict]]], outdir: Path):
    all_results = [r for _, res in sources for r in res]
    unique_widths = sorted(set(r["width"] for r in all_results))
    nrows = len(sources)
    ncols = len(unique_widths)
    fig, axes = _subplots_with_thumbnails(sources, ncols, col_width=3.5, row_height=3.5)

    for row, (src_name, results) in enumerate(sources):
        for col, w in enumerate(unique_widths):
            ax = axes[row][col]
            for fmt in formats_in(results):
                pts = sorted(
                    [r for r in results if r["format"] == fmt and r["width"] == w],
                    key=lambda r: r["quality"],
                )
                if not pts:
                    continue
                ax.plot(
                    [r["size_kb"] for r in pts], [m(r) for r in pts],
                    c=FORMAT_COLORS[fmt], marker="o", markersize=3,
                    label=FORMAT_LABELS[fmt] if row == 0 and col == 0 else None,
                    alpha=0.8, linewidth=1.2,
                )
                if fmt == "avif":
                    for r in pts:
                        ax.annotate(
                            f"q{r['quality']}", (r["size_kb"], m(r)),
                            fontsize=5, alpha=0.5, color=FORMAT_COLORS["avif"],
                            xytext=(3, 3), textcoords="offset points",
                        )
            ax.axhline(y=70, color="#e0e0e0", linestyle="--", linewidth=0.7)
            ax.set_xscale("log", base=2)
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(
                lambda x, _: f"{x/1024:.0f}MB" if x >= 1024 else f"{x:.0f}KB"
            ))
            if row == 0:
                ax.set_title(f"{w}px")
            if col == 0:
                ax.set_ylabel(f"{src_name}\n{metric_label()}")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles),
                   bbox_to_anchor=(0.5, 1.0))

    fig.supxlabel("File Size (KB)")
    fig.suptitle(f"{metric_label()} by Resolution", y=1.02)
    save_fig(fig, outdir, "pareto-by-width.png")


# ---------------------------------------------------------------------------
# By-quality: faceted by quality level, rows = source images
# ---------------------------------------------------------------------------

def plot_by_quality(sources: list[tuple[str, list[dict]]], outdir: Path):
    all_results = [r for _, res in sources for r in res]
    all_qualities = sorted(set(r["quality"] for r in all_results))
    # Show a subset to keep it readable
    if len(all_qualities) > 5:
        idxs = np.linspace(0, len(all_qualities) - 1, 5, dtype=int)
        show_qualities = [all_qualities[i] for i in idxs]
    else:
        show_qualities = all_qualities

    nrows = len(sources)
    ncols = len(show_qualities)
    fig, axes = _subplots_with_thumbnails(sources, ncols, col_width=3.5, row_height=3.5)

    for row, (src_name, results) in enumerate(sources):
        for col, q in enumerate(show_qualities):
            ax = axes[row][col]
            for fmt in formats_in(results):
                pts = sorted(
                    [r for r in results if r["format"] == fmt and r["quality"] == q],
                    key=lambda r: r["width"],
                )
                if not pts:
                    continue
                ax.plot(
                    [r["size_kb"] for r in pts], [m(r) for r in pts],
                    c=FORMAT_COLORS[fmt], marker="o", markersize=3,
                    label=FORMAT_LABELS[fmt] if row == 0 and col == 0 else None,
                    alpha=0.8, linewidth=1.2,
                )
                # Label endpoints with resolution
                for r in [pts[0], pts[-1]]:
                    ax.annotate(
                        f"{r['width']}px", (r["size_kb"], m(r)),
                        fontsize=5, alpha=0.5, xytext=(3, 3), textcoords="offset points",
                    )
            ax.axhline(y=70, color="#e0e0e0", linestyle="--", linewidth=0.7)
            if row == 0:
                ax.set_title(f"q{q}")
            if col == 0:
                ax.set_ylabel(f"{src_name}\n{metric_label()}")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles),
                   bbox_to_anchor=(0.5, 1.0))

    fig.supxlabel("File Size (KB)")
    fig.suptitle(f"Resolution Impact at Fixed Quality ({metric_label()})", y=1.02)
    save_fig(fig, outdir, "pareto-by-quality.png")


# ---------------------------------------------------------------------------
# Sensitivity: quality, resolution, and marginal efficiency
# ---------------------------------------------------------------------------

def plot_sensitivity(sources: list[tuple[str, list[dict]]], outdir: Path):
    nrows = len(sources)
    fig, axes = _subplots_with_thumbnails(sources, 3, col_width=5.3, row_height=5)

    mid_qualities = {"avif": 60, "jxl": 60, "webp": 70, "jpeg": 75}

    for row, (src_name, results) in enumerate(sources):
        fmts = formats_in(results)

        # Left: SSIM vs quality at 1600px
        ax = axes[row][0]
        for fmt in fmts:
            pts = sorted(
                [r for r in results if r["format"] == fmt and r["width"] == 1600],
                key=lambda r: r["quality"],
            )
            if pts:
                ax.plot(
                    [r["quality"] for r in pts], [m(r) for r in pts],
                    color=FORMAT_COLORS[fmt], marker="o", markersize=5,
                    label=FORMAT_LABELS[fmt], linewidth=1.5,
                )
        ax.set_xlabel("Quality Setting")
        ax.set_ylabel(metric_label())
        ax.set_title(f"{src_name}\n{metric_label()} vs Quality (1600px)")

        # Middle: SSIM vs resolution at mid quality
        ax = axes[row][1]
        for fmt in fmts:
            mq = mid_qualities.get(fmt)
            if not mq:
                continue
            pts = sorted(
                [r for r in results if r["format"] == fmt and r["quality"] == mq],
                key=lambda r: r["width"],
            )
            if pts:
                ax.plot(
                    [r["width"] for r in pts], [m(r) for r in pts],
                    color=FORMAT_COLORS[fmt], marker="o", markersize=5,
                    label=f"{FORMAT_LABELS[fmt]} q{mq}", linewidth=1.5,
                )
        ax.set_xlabel("Image Width (px)")
        ax.set_ylabel(metric_label())
        ax.set_title(f"{src_name}\n{metric_label()} vs Resolution (mid quality)")

        # Right: marginal efficiency
        ax = axes[row][2]
        for fi, fmt in enumerate(fmts):
            fmt_pts = [r for r in results if r["format"] == fmt]
            if not fmt_pts:
                continue
            widths_seen = sorted(set(r["width"] for r in fmt_pts))
            qs_seen = sorted(set(r["quality"] for r in fmt_pts))

            eq = []
            for w in widths_seen:
                pts = sorted([r for r in fmt_pts if r["width"] == w], key=lambda r: r["quality"])
                for j in range(1, len(pts)):
                    ds = pts[j]["size_kb"] - pts[j - 1]["size_kb"]
                    if ds > 0:
                        eq.append((m(pts[j]) - m(pts[j-1])) / ds * 100)

            ew = []
            for q in qs_seen:
                pts = sorted([r for r in fmt_pts if r["quality"] == q], key=lambda r: r["width"])
                for j in range(1, len(pts)):
                    ds = pts[j]["size_kb"] - pts[j - 1]["size_kb"]
                    if ds > 0:
                        ew.append((m(pts[j]) - m(pts[j-1])) / ds * 100)

            ax.bar(fi - 0.15, np.mean(eq) if eq else 0, 0.28,
                   color=FORMAT_COLORS[fmt], alpha=0.8,
                   label="Quality" if fi == 0 else None)
            ax.bar(fi + 0.15, np.mean(ew) if ew else 0, 0.28,
                   color=FORMAT_COLORS[fmt], alpha=0.4,
                   label="Resolution" if fi == 0 else None,
                   hatch="//", edgecolor=FORMAT_COLORS[fmt])

        ax.set_xticks(range(len(fmts)))
        ax.set_xticklabels([FORMAT_LABELS[f] for f in fmts])
        ax.set_ylabel(f"Avg d{metric_label()} per 100KB")
        ax.set_title(f"{src_name}\nMarginal Efficiency")

    fig.suptitle("Sensitivity Analysis")
    save_fig(fig, outdir, "sensitivity.png")


# ---------------------------------------------------------------------------
# Iso-SSIM: hold SSIM constant, show file size cost
# ---------------------------------------------------------------------------

def plot_iso_ssim(sources: list[tuple[str, list[dict]]], outdir: Path):
    targets = [50, 70, 80]
    nrows = len(sources)
    ncols = len(targets)
    fig, axes = _subplots_with_thumbnails(sources, ncols, col_width=5, row_height=4.5)

    for row, (src_name, results) in enumerate(sources):
        for ti, target in enumerate(targets):
            ax = axes[row][ti]
            for fmt in formats_in(results):
                widths_seen = sorted(set(r["width"] for r in results if r["format"] == fmt))
                iso_widths, iso_sizes = [], []
                for w in widths_seen:
                    pts = sorted(
                        [r for r in results if r["format"] == fmt and r["width"] == w],
                        key=lambda r: r["quality"],
                    )
                    ssims = [m(r) for r in pts]
                    sizes = [r["size_kb"] for r in pts]
                    if min(ssims) <= target <= max(ssims):
                        try:
                            f_interp = interpolate.interp1d(ssims, sizes, kind="linear")
                            iso_widths.append(w)
                            iso_sizes.append(float(f_interp(target)))
                        except ValueError:
                            pass
                if iso_widths:
                    ax.plot(
                        iso_widths, iso_sizes,
                        color=FORMAT_COLORS[fmt], marker="o", markersize=5,
                        label=FORMAT_LABELS[fmt], linewidth=1.5,
                    )
            ax.set_xlabel("Image Width (px)")
            if ti == 0:
                ax.set_ylabel(f"{src_name}\nFile Size (KB)")
            if row == 0:
                ax.set_title(f"{metric_label()} = {target}")

    fig.suptitle(f"Iso-{metric_label()}: Cost to Achieve Target Quality")
    save_fig(fig, outdir, "iso-ssim.png")


# ---------------------------------------------------------------------------
# Iso-size: hold file size constant, show best SSIM
# ---------------------------------------------------------------------------

def plot_iso_size(sources: list[tuple[str, list[dict]]], outdir: Path):
    targets = [50, 100, 200, 400, 800]
    nrows = len(sources)
    ncols = len(targets)
    fig, axes = _subplots_with_thumbnails(sources, ncols, col_width=3.5, row_height=4.5)

    for row, (src_name, results) in enumerate(sources):
        for ti, target_kb in enumerate(targets):
            ax = axes[row][ti]
            for fmt in formats_in(results):
                widths_seen = sorted(set(r["width"] for r in results if r["format"] == fmt))
                iso_widths, iso_ssims = [], []
                for w in widths_seen:
                    pts = sorted(
                        [r for r in results if r["format"] == fmt and r["width"] == w],
                        key=lambda r: r["size_kb"],
                    )
                    sizes = [r["size_kb"] for r in pts]
                    ssims = [m(r) for r in pts]
                    if min(sizes) <= target_kb <= max(sizes):
                        try:
                            f_interp = interpolate.interp1d(sizes, ssims, kind="linear")
                            iso_widths.append(w)
                            iso_ssims.append(float(f_interp(target_kb)))
                        except ValueError:
                            pass
                    elif max(sizes) < target_kb:
                        best_pt = max(pts, key=lambda r: m(r))
                        iso_widths.append(w)
                        iso_ssims.append(m(best_pt))
                if iso_widths:
                    ax.plot(
                        iso_widths, iso_ssims,
                        color=FORMAT_COLORS[fmt], marker="o", markersize=4,
                        label=FORMAT_LABELS[fmt], linewidth=1.3, alpha=0.7,
                    )
            ax.set_xlabel("Image Width (px)")
            if ti == 0:
                ax.set_ylabel(f"{src_name}\nBest {metric_label()}")
            if row == 0:
                ax.set_title(f"{target_kb} KB")

    fig.suptitle(f"Iso-Size: Best {metric_label()} at a Given File Size")
    save_fig(fig, outdir, "iso-size.png")


# ---------------------------------------------------------------------------
# Heatmaps: quality x width grid per format, rows = source images
# ---------------------------------------------------------------------------

def plot_size_heatmaps(sources: list[tuple[str, list[dict]]], outdir: Path):
    all_results = [r for _, res in sources for r in res]
    fmts = formats_in(all_results)
    nrows = len(sources)
    ncols = len(fmts)

    # Size heatmaps
    fig, axes = _subplots_with_thumbnails(sources, ncols, col_width=4, row_height=3.5)
    for row, (src_name, results) in enumerate(sources):
        for col, fmt in enumerate(fmts):
            ax = axes[row][col]
            ax.grid(False)
            pts = [r for r in results if r["format"] == fmt]
            qs = sorted(set(r["quality"] for r in pts))
            ws = sorted(set(r["width"] for r in pts))
            grid = np.full((len(qs), len(ws)), np.nan)
            for r in pts:
                grid[qs.index(r["quality"])][ws.index(r["width"])] = r["size_kb"]
            im = ax.pcolormesh(grid, cmap="YlOrRd", edgecolors="face", linewidth=0.5, antialiased=True)
            ax.set_xticks(np.arange(len(ws)) + 0.5)
            ax.set_xticklabels([str(w) for w in ws], fontsize=6, rotation=45)
            ax.set_yticks(np.arange(len(qs)) + 0.5)
            ax.set_yticklabels([f"q{q}" for q in qs], fontsize=6)
            ax.set_xlim(0, len(ws))
            ax.set_ylim(0, len(qs))
            for qi in range(len(qs)):
                for wi in range(len(ws)):
                    v = grid[qi][wi]
                    if not np.isnan(v):
                        ax.text(wi + 0.5, qi + 0.5, f"{v:.0f}", ha="center", va="center", fontsize=5,
                                color="white" if v > np.nanmax(grid) * 0.6 else "black")
            fig.colorbar(im, ax=ax, shrink=0.8)
            if row == 0:
                ax.set_title(FORMAT_LABELS[fmt])
            if col == 0:
                ax.set_ylabel(f"{src_name}\nQuality")

    fig.suptitle("File Size (KB) by Quality x Resolution")
    save_fig(fig, outdir, "heatmap-size.png")

    # SSIM heatmaps
    fig, axes = _subplots_with_thumbnails(sources, ncols, col_width=4, row_height=3.5)
    for row, (src_name, results) in enumerate(sources):
        for col, fmt in enumerate(fmts):
            ax = axes[row][col]
            ax.grid(False)
            pts = [r for r in results if r["format"] == fmt]
            qs = sorted(set(r["quality"] for r in pts))
            ws = sorted(set(r["width"] for r in pts))
            grid = np.full((len(qs), len(ws)), np.nan)
            for r in pts:
                grid[qs.index(r["quality"])][ws.index(r["width"])] = m(r)
            im = ax.pcolormesh(grid, cmap="RdYlGn", edgecolors="face", linewidth=0.5, antialiased=True, vmax=100)
            ax.set_xticks(np.arange(len(ws)) + 0.5)
            ax.set_xticklabels([str(w) for w in ws], fontsize=6, rotation=45)
            ax.set_yticks(np.arange(len(qs)) + 0.5)
            ax.set_yticklabels([f"q{q}" for q in qs], fontsize=6)
            ax.set_xlim(0, len(ws))
            ax.set_ylim(0, len(qs))
            val_fmt = ".1f"
            low_thresh = 50
            for qi in range(len(qs)):
                for wi in range(len(ws)):
                    v = grid[qi][wi]
                    if not np.isnan(v):
                        ax.text(wi + 0.5, qi + 0.5, f"{v:{val_fmt}}", ha="center", va="center", fontsize=5,
                                color="white" if v < low_thresh else "black")
            fig.colorbar(im, ax=ax, shrink=0.8)
            if row == 0:
                ax.set_title(FORMAT_LABELS[fmt])
            if col == 0:
                ax.set_ylabel(f"{src_name}\nQuality")

    fig.suptitle(f"{metric_label()} by Quality x Resolution")
    save_fig(fig, outdir, "heatmap-ssim.png")


# ---------------------------------------------------------------------------
# Contour: SSIM surface with file size overlay, per format
# ---------------------------------------------------------------------------

def plot_contour(sources: list[tuple[str, list[dict]]], outdir: Path):
    all_results = [r for _, res in sources for r in res]
    fmts = formats_in(all_results)
    nrows = len(sources)
    ncols = len(fmts)
    fig, axes = _subplots_with_thumbnails(sources, ncols, col_width=4.5, row_height=4)

    metric_levels = [10, 30, 50, 60, 70, 80, 85, 90]
    fmt_str = "%.0f"

    for row, (src_name, results) in enumerate(sources):
        for col, fmt in enumerate(fmts):
            ax = axes[row][col]
            ax.grid(False)
            pts = [r for r in results if r["format"] == fmt]
            if not pts:
                continue

            qs = sorted(set(r["quality"] for r in pts))
            ws = sorted(set(r["width"] for r in pts))

            metric_grid = np.full((len(qs), len(ws)), np.nan)
            size_grid = np.full((len(qs), len(ws)), np.nan)
            for r in pts:
                qi = qs.index(r["quality"])
                wi = ws.index(r["width"])
                metric_grid[qi][wi] = m(r)
                size_grid[qi][wi] = r["size_kb"]

            from scipy.ndimage import zoom
            metric_smooth = zoom(metric_grid, (100 / len(qs), 100 / len(ws)), order=3)
            size_smooth = zoom(size_grid, (100 / len(qs), 100 / len(ws)), order=3)

            q_vals = np.linspace(qs[0], qs[-1], metric_smooth.shape[0])
            w_vals = np.linspace(ws[0], ws[-1], metric_smooth.shape[1])
            W, Q = np.meshgrid(w_vals, q_vals)

            # Color = file size
            pcm = ax.pcolormesh(W, Q, size_smooth, cmap="YlOrRd", alpha=0.6, shading="auto")

            # Contour lines = metric levels
            cs = ax.contour(W, Q, metric_smooth, levels=metric_levels,
                            colors="black", linewidths=0.8, alpha=0.7)
            ax.clabel(cs, inline=True, fontsize=7, fmt=fmt_str)

            # Scatter the actual data points
            for r in pts:
                ax.plot(r["width"], r["quality"], "k.", markersize=1.5, alpha=0.3)

            ax.set_xlabel("Width (px)")
            ax.set_ylabel("Quality")

            if row == 0:
                ax.set_title(FORMAT_LABELS[fmt])
            if col == 0:
                ax.set_ylabel(f"{src_name}\nQuality")

            fig.colorbar(pcm, ax=ax, shrink=0.8, label="Size (KB)")

    fig.suptitle(f"{metric_label()} Contours over File Size (color)")
    save_fig(fig, outdir, "contour.png")


def plot_contour_inv(sources: list[tuple[str, list[dict]]], outdir: Path):
    """Inverted contour: SSIM as color, file size as contour lines."""
    all_results = [r for _, res in sources for r in res]
    fmts = formats_in(all_results)
    nrows = len(sources)
    ncols = len(fmts)
    fig, axes = _subplots_with_thumbnails(sources, ncols, col_width=4.5, row_height=4)

    from scipy.ndimage import zoom

    vmin, vmax = -10, 100

    for row, (src_name, results) in enumerate(sources):
        for col, fmt in enumerate(fmts):
            ax = axes[row][col]
            ax.grid(False)
            pts = [r for r in results if r["format"] == fmt]
            if not pts:
                continue

            qs = sorted(set(r["quality"] for r in pts))
            ws = sorted(set(r["width"] for r in pts))

            metric_grid = np.full((len(qs), len(ws)), np.nan)
            size_grid = np.full((len(qs), len(ws)), np.nan)
            for r in pts:
                qi = qs.index(r["quality"])
                wi = ws.index(r["width"])
                metric_grid[qi][wi] = m(r)
                size_grid[qi][wi] = r["size_kb"]

            metric_smooth = zoom(metric_grid, (100 / len(qs), 100 / len(ws)), order=3)
            size_smooth = zoom(size_grid, (100 / len(qs), 100 / len(ws)), order=3)

            q_vals = np.linspace(qs[0], qs[-1], metric_smooth.shape[0])
            w_vals = np.linspace(ws[0], ws[-1], metric_smooth.shape[1])
            W, Q = np.meshgrid(w_vals, q_vals)

            # Color = metric value
            pcm = ax.pcolormesh(W, Q, metric_smooth, cmap="RdYlGn", alpha=0.7,
                                shading="auto", vmin=vmin, vmax=vmax)

            # Contour lines = file size
            size_levels = [50, 100, 200, 400, 800, 1500]
            cs = ax.contour(W, Q, size_smooth, levels=size_levels,
                            colors="black", linewidths=0.8, alpha=0.7)
            ax.clabel(cs, inline=True, fontsize=7, fmt="%dKB")

            for r in pts:
                ax.plot(r["width"], r["quality"], "k.", markersize=1.5, alpha=0.3)

            ax.set_xlabel("Width (px)")
            ax.set_ylabel("Quality")

            if row == 0:
                ax.set_title(FORMAT_LABELS[fmt])
            if col == 0:
                ax.set_ylabel(f"{src_name}\nQuality")

            fig.colorbar(pcm, ax=ax, shrink=0.8, label=metric_label())

    fig.suptitle(f"File Size Contours over {metric_label()} (color)")
    save_fig(fig, outdir, "contour-inv.png")


# ---------------------------------------------------------------------------
# Efficiency contour: SSIM-per-KB as color, showing the sweet spot
# ---------------------------------------------------------------------------

def plot_efficiency(sources: list[tuple[str, list[dict]]], outdir: Path):
    """Efficiency landscape: gradient magnitude of SSIM w.r.t. file size."""
    all_results = [r for _, res in sources for r in res]
    fmts = formats_in(all_results)
    nrows = len(sources)
    ncols = len(fmts)
    fig, axes = _subplots_with_thumbnails(sources, ncols, col_width=4.5, row_height=4)

    from scipy.ndimage import zoom

    for row, (src_name, results) in enumerate(sources):
        for col, fmt in enumerate(fmts):
            ax = axes[row][col]
            ax.grid(False)
            pts = [r for r in results if r["format"] == fmt]
            if not pts:
                continue

            qs = sorted(set(r["quality"] for r in pts))
            ws = sorted(set(r["width"] for r in pts))

            metric_grid = np.full((len(qs), len(ws)), np.nan)
            size_grid = np.full((len(qs), len(ws)), np.nan)
            for r in pts:
                qi = qs.index(r["quality"])
                wi = ws.index(r["width"])
                metric_grid[qi][wi] = m(r)
                size_grid[qi][wi] = r["size_kb"]

            # Compute efficiency: delta-SSIM / delta-size along quality axis
            eff_grid = np.full_like(metric_grid, np.nan)
            for wi in range(len(ws)):
                for qi in range(1, len(qs)):
                    ds = size_grid[qi, wi] - size_grid[qi - 1, wi]
                    dm = metric_grid[qi, wi] - metric_grid[qi - 1, wi]
                    if ds > 0:
                        eff_grid[qi, wi] = dm / ds * 100  # per 100KB
                # Fill first row with second row value
                if not np.isnan(eff_grid[1, wi]):
                    eff_grid[0, wi] = eff_grid[1, wi]

            eff_smooth = zoom(eff_grid, (100 / len(qs), 100 / len(ws)), order=3)
            size_smooth = zoom(size_grid, (100 / len(qs), 100 / len(ws)), order=3)

            q_vals = np.linspace(qs[0], qs[-1], eff_smooth.shape[0])
            w_vals = np.linspace(ws[0], ws[-1], eff_smooth.shape[1])
            W, Q = np.meshgrid(w_vals, q_vals)

            # Clip efficiency to reasonable range for color mapping
            vmax = np.nanpercentile(eff_grid, 95)
            pcm = ax.pcolormesh(W, Q, np.clip(eff_smooth, 0, vmax),
                                cmap="YlGn", alpha=0.8, shading="auto")

            # Contour lines = file size
            size_levels = [50, 100, 200, 400, 800, 1500]
            cs = ax.contour(W, Q, size_smooth, levels=size_levels,
                            colors="black", linewidths=0.8, alpha=0.5)
            ax.clabel(cs, inline=True, fontsize=7, fmt="%dKB")

            for r in pts:
                ax.plot(r["width"], r["quality"], "k.", markersize=1.5, alpha=0.3)

            ax.set_xlabel("Width (px)")
            ax.set_ylabel("Quality")

            if row == 0:
                ax.set_title(FORMAT_LABELS[fmt])
            if col == 0:
                ax.set_ylabel(f"{src_name}\nQuality")

            fig.colorbar(pcm, ax=ax, shrink=0.8,
                         label=f"d{metric_label()} per 100KB")

    fig.suptitle(f"Efficiency: {metric_label()} Gain per 100KB (quality axis)")
    save_fig(fig, outdir, "efficiency.png")


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def generate_all(results: list[dict], outdir: Path, metric: str | None = None,
                  source_dir: Path | None = None):
    global METRIC, SOURCE_DIR
    init_style()
    if metric is not None:
        METRIC = metric
    SOURCE_DIR = source_dir
    sources = split_by_source(results)
    plot_pareto(sources, outdir)
    plot_by_width(sources, outdir)
    plot_by_quality(sources, outdir)
    plot_sensitivity(sources, outdir)
    plot_iso_ssim(sources, outdir)
    plot_iso_size(sources, outdir)
    plot_size_heatmaps(sources, outdir)
    plot_contour(sources, outdir)
    plot_contour_inv(sources, outdir)
    plot_knob_landscape(sources, outdir)
    plot_efficiency(sources, outdir)


# ---------------------------------------------------------------------------
# Knob landscape: for best format, show resolution x quality → size vs score
# with Pareto frontier
# ---------------------------------------------------------------------------

def _pareto_frontier(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Given (size, score) pairs, return the Pareto-optimal subset (min size, max score)."""
    sorted_pts = sorted(points, key=lambda p: p[0])
    frontier = []
    best_score = float("-inf")
    for size, score in sorted_pts:
        if score > best_score:
            frontier.append((size, score))
            best_score = score
    return frontier


def plot_knob_landscape(sources: list[tuple[str, list[dict]]], outdir: Path, fmt: str = "avif"):
    nrows = len(sources)
    fig, axes = _subplots_with_thumbnails(sources, 1, col_width=10, row_height=7)

    for row, (src_name, results) in enumerate(sources):
        pts = [r for r in results if r["format"] == fmt]
        if not pts:
            continue

        ax = axes[row][0]
        widths = sorted(set(r["width"] for r in pts))

        # Soft background curves per resolution, faded
        cmap = plt.cm.cool(np.linspace(0.15, 0.85, len(widths)))
        all_points = []
        for i, w in enumerate(widths):
            w_pts = sorted([r for r in pts if r["width"] == w], key=lambda r: r["quality"])
            sizes = [r["size_kb"] for r in w_pts]
            scores = [m(r) for r in w_pts]
            ax.plot(sizes, scores, color=cmap[i], marker=".", markersize=3,
                    linewidth=0.9, alpha=0.35)
            # Small label at the high-quality end of each curve
            best = w_pts[-1]
            ax.annotate(f"{w}px", (best["size_kb"], m(best)),
                        fontsize=6.5, color=cmap[i], alpha=0.7,
                        xytext=(5, 0), textcoords="offset points", va="center")
            all_points.extend((r["size_kb"], m(r), r) for r in w_pts)

        # Pareto frontier as the hero element
        frontier = _pareto_frontier([(s, sc) for s, sc, _ in all_points])
        if frontier:
            f_sizes = [p[0] for p in frontier]
            f_scores = [p[1] for p in frontier]
            ax.fill_between(f_sizes, f_scores, min(f_scores) - 5,
                            color="#2a9d8f", alpha=0.08)
            ax.plot(f_sizes, f_scores,
                    color="#2a9d8f", linewidth=2.5, alpha=0.9,
                    zorder=10, label="Pareto frontier")
            ax.scatter(f_sizes, f_scores, color="#2a9d8f", s=25,
                       zorder=11, edgecolors="white", linewidths=0.5)

            # Label frontier points, spaced to avoid overlap
            frontier_set = set(frontier)
            frontier_results = sorted(
                [(s, sc, r) for s, sc, r in all_points if (s, sc) in frontier_set],
                key=lambda x: x[0],
            )
            n_labels = min(6, len(frontier_results))
            if n_labels > 0:
                idxs = np.linspace(0, len(frontier_results) - 1, n_labels, dtype=int)
                for idx in idxs:
                    s, sc, r = frontier_results[idx]
                    ax.annotate(
                        f"{r['width']}px  q{r['quality']}",
                        (s, sc), fontsize=7,
                        xytext=(8, -12), textcoords="offset points",
                        arrowprops=dict(arrowstyle="-", color="#2a9d8f", lw=0.7),
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                  edgecolor="#2a9d8f", alpha=0.9, linewidth=0.7),
                        color="#264653",
                    )

        add_quality_bands(ax, side="right")

        ax.set_xlabel("File Size (KB)")
        ax.set_ylabel(metric_label())
        ax.set_title(src_name)
        ax.legend(loc="lower right")

    fig.suptitle(f"{FORMAT_LABELS[fmt]}: Resolution x Quality Landscape", y=1.01)
    save_fig(fig, outdir, "knob-landscape.png")


def main():
    global METRIC
    parser = argparse.ArgumentParser(description="Generate plots from benchmark CSV")
    parser.add_argument("csv", type=Path, help="Path to results.csv")
    parser.add_argument("-o", "--outdir", type=Path, default=None,
                        help="Output directory (default: same dir as CSV)")
    parser.add_argument("-m", "--metric", default="ssimulacra2",
                        choices=list(METRIC_LABELS.keys()),
                        help="Metric to plot (default: ssimulacra2)")
    parser.add_argument("-s", "--source-dir", type=Path, default=None,
                        help="Directory containing source images (for thumbnails)")
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"CSV not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    outdir = args.outdir or args.csv.parent
    outdir.mkdir(parents=True, exist_ok=True)

    results = load_results(args.csv)
    print(f"Loaded {len(results)} results from {args.csv}")
    print(f"Plotting metric: {metric_label()}")
    generate_all(results, outdir, metric=args.metric, source_dir=args.source_dir)


if __name__ == "__main__":
    main()
