import argparse
import html
import math
import re
from pathlib import Path


MODEL_RE = re.compile(r"^Model:\s*(.+?)\s*$", re.M)
BACKEND_RE = re.compile(r"^Backend:\s*(.+?)\s*$", re.M)
EXIT_RE = re.compile(r"^Exit code:\s*(-?\d+)\s*$", re.M)
TRANSCRIBE_RE = re.compile(r"^Transcribe time:\s*([0-9]+(?:\.[0-9]+)?)s\s*$", re.M)
REPEATS_RE = re.compile(r"^Repeats:\s*(\d+)\s*$", re.M)


def parse_results(text):
    records = []
    repeat_matches = REPEATS_RE.findall(text)
    repeats = max((int(x) for x in repeat_matches), default=None)
    for block in text.split("\n\n"):
        if "Model:" not in block:
            continue
        model_match = MODEL_RE.search(block)
        if not model_match:
            continue
        backend_matches = BACKEND_RE.findall(block)
        backend = backend_matches[-1].strip() if backend_matches else "unknown"
        exit_match = EXIT_RE.search(block)
        exit_code = int(exit_match.group(1)) if exit_match else None
        transcribe_match = TRANSCRIBE_RE.search(block)
        transcribe_time = float(transcribe_match.group(1)) if transcribe_match else None
        records.append(
            {
                "model": model_match.group(1).strip(),
                "backend": backend,
                "exit_code": exit_code,
                "transcribe_time": transcribe_time,
            }
        )
    return records, repeats


def pick_tick_step(max_value):
    if max_value <= 1.0:
        return 0.1
    rough = max_value / 6.0
    power = 10 ** math.floor(math.log10(rough))
    for m in (1, 2, 5, 10):
        step = m * power
        if step >= rough:
            return step
    return 10 * power


def color_for_backend(backend):
    backend = (backend or "").lower()
    if backend == "hf":
        return "#1f77b4"
    if backend == "faster":
        return "#ff7f0e"
    return "#6b7280"


def build_svg(records, title, repeats=None):
    successful = [r for r in records if r["transcribe_time"] is not None]
    failed = [r for r in records if r["transcribe_time"] is None]
    successful = sorted(
        successful,
        key=lambda r: (float(r["transcribe_time"]), str(r["backend"]).lower(), str(r["model"]).lower()),
    )

    subtitle = "Transcribe time in seconds (lower is better) | sorted: fastest -> slowest"
    if repeats is not None:
        subtitle = f"{subtitle} | repeats per model: {repeats}"

    if not successful:
        width = 1200
        height = 250 + (18 * len(failed))
        parts = []
        add = parts.append
        add(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">'
        )
        add(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>')
        add(
            f'<text x="40" y="42" font-family="Segoe UI, Arial, sans-serif" '
            f'font-size="24" font-weight="700" fill="#111827">{html.escape(title)}</text>'
        )
        add(
            f'<text x="40" y="68" font-family="Segoe UI, Arial, sans-serif" '
            f'font-size="13" fill="#4b5563">{html.escape(subtitle)}</text>'
        )
        add(
            f'<text x="40" y="110" font-family="Segoe UI, Arial, sans-serif" '
            f'font-size="16" fill="#b91c1c">No successful transcribe times found.</text>'
        )
        if failed:
            add(
                f'<text x="40" y="138" font-family="Segoe UI, Arial, sans-serif" '
                f'font-size="12" fill="#b91c1c">Failed entries:</text>'
            )
            for i, rec in enumerate(failed):
                add(
                    f'<text x="56" y="{156 + i * 18}" '
                    f'font-family="Consolas, monospace" font-size="11" fill="#b91c1c">- '
                    f'{html.escape(rec["backend"])} | {html.escape(rec["model"])} (exit={rec["exit_code"]})</text>'
                )
        add("</svg>")
        return "\n".join(parts)

    labels = [f"{r['backend']} | {r['model']}" for r in successful]
    max_label_chars = max(len(lbl) for lbl in labels)

    row_h = 34
    chart_top = 70
    left_margin = min(640, 120 + max_label_chars * 7)
    right_margin = 120
    chart_w = 980
    chart_h = row_h * len(successful)
    foot_h = 24 + (18 * len(failed) if failed else 0)
    width = left_margin + chart_w + right_margin
    height = chart_top + chart_h + 70 + foot_h

    max_time = max(r["transcribe_time"] for r in successful)
    x_max = max_time * 1.15
    tick_step = pick_tick_step(x_max)
    tick_max = math.ceil(x_max / tick_step) * tick_step

    parts = []
    add = parts.append
    add(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">'
    )
    add(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>')
    add(
        f'<text x="{left_margin}" y="34" font-family="Segoe UI, Arial, sans-serif" '
        f'font-size="24" font-weight="700" fill="#111827">{html.escape(title)}</text>'
    )
    add(
        f'<text x="{left_margin}" y="56" font-family="Segoe UI, Arial, sans-serif" '
        f'font-size="13" fill="#4b5563">{html.escape(subtitle)}</text>'
    )

    # Grid + ticks
    tick_count = int(round(tick_max / tick_step))
    for i in range(tick_count + 1):
        val = i * tick_step
        x = left_margin + (val / tick_max) * chart_w
        add(f'<line x1="{x:.2f}" y1="{chart_top}" x2="{x:.2f}" y2="{chart_top + chart_h}" stroke="#e5e7eb" stroke-width="1"/>')
        add(
            f'<text x="{x:.2f}" y="{chart_top + chart_h + 20}" text-anchor="middle" '
            f'font-family="Consolas, monospace" font-size="12" fill="#6b7280">{val:.1f}s</text>'
        )

    # Bars
    for i, rec in enumerate(successful):
        y = chart_top + i * row_h + 5
        bar_h = row_h - 10
        value = rec["transcribe_time"]
        bar_w = (value / tick_max) * chart_w
        label = f"{rec['backend']} | {rec['model']}"
        color = color_for_backend(rec["backend"])

        add(
            f'<text x="{left_margin - 10}" y="{y + bar_h * 0.72:.2f}" text-anchor="end" '
            f'font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#111827">{html.escape(label)}</text>'
        )
        add(f'<rect x="{left_margin}" y="{y}" width="{bar_w:.2f}" height="{bar_h}" rx="4" fill="{color}"/>')
        add(
            f'<text x="{left_margin + bar_w + 8:.2f}" y="{y + bar_h * 0.72:.2f}" '
            f'font-family="Consolas, monospace" font-size="12" fill="#111827">{value:.2f}s</text>'
        )

    legend_y = chart_top + chart_h + 48
    add(f'<rect x="{left_margin}" y="{legend_y - 12}" width="14" height="14" fill="#1f77b4"/>')
    add(
        f'<text x="{left_margin + 20}" y="{legend_y}" font-family="Segoe UI, Arial, sans-serif" '
        f'font-size="12" fill="#111827">hf</text>'
    )
    add(f'<rect x="{left_margin + 70}" y="{legend_y - 12}" width="14" height="14" fill="#ff7f0e"/>')
    add(
        f'<text x="{left_margin + 90}" y="{legend_y}" font-family="Segoe UI, Arial, sans-serif" '
        f'font-size="12" fill="#111827">faster</text>'
    )

    if failed:
        add(
            f'<text x="{left_margin}" y="{legend_y + 24}" font-family="Segoe UI, Arial, sans-serif" '
            f'font-size="12" fill="#b91c1c">Skipped (missing transcribe time):</text>'
        )
        for i, rec in enumerate(failed):
            add(
                f'<text x="{left_margin + 12}" y="{legend_y + 42 + i * 18}" '
                f'font-family="Consolas, monospace" font-size="11" fill="#b91c1c">- '
                f'{html.escape(rec["backend"])} | {html.escape(rec["model"])} (exit={rec["exit_code"]})</text>'
            )

    add("</svg>")
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser(description="Generate an SVG bar chart of transcribe times.")
    ap.add_argument("--input", default="performance_results.txt", help="Path to performance results text file")
    ap.add_argument("--output", default="transcribe_times.svg", help="Output SVG file path")
    ap.add_argument("--title", default="Whisper Transcribe Time Comparison", help="Chart title")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    text = in_path.read_text(encoding="utf-8", errors="replace")
    records, repeats = parse_results(text)
    svg = build_svg(records, args.title, repeats=repeats)
    out_path.write_text(svg, encoding="utf-8")
    print(f"Wrote chart: {out_path}")


if __name__ == "__main__":
    main()
