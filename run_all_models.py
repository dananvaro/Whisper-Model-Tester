import argparse
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

HF_MODELS = [
    "openai/whisper-tiny",
    "openai/whisper-small",
    "openai/whisper-base",
    "openai/whisper-medium",
    "openai/whisper-large",
    "NbAiLab/nb-whisper-tiny",
    "NbAiLab/nb-whisper-small",
    "NbAiLab/nb-whisper-base",
    "NbAiLab/nb-whisper-medium",
    "NbAiLab/nb-whisper-large",
]

FASTER_MODELS = [
    "tiny",
    "small",
    "base",
    "medium",
    "large-v3",
]

SCRIPT_BY_BACKEND = {
    "hf": "resultsWhisper.py",
    "faster": "resultsFasterWhisper.py",
}

ERROR_HINT_PATTERNS = [
    re.compile(r"\bRetrying in\b", re.I),
    re.compile(r"\bConnectionError\b", re.I),
    re.compile(r"\bFailed to connect\b", re.I),
    re.compile(r"\bOSError:\b", re.I),
    re.compile(r"\bRuntimeError:\b", re.I),
    re.compile(r"\bTraceback\b", re.I),
]

PROXY_ENV_KEYS = [
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
]

RUN_METRIC_PATTERNS = {
    "device": re.compile(r"^Device:\s*(.+?)\s*$", re.M),
    "compute_type": re.compile(r"^Compute type:\s*(.+?)\s*$", re.M),
    "load_time": re.compile(r"^Load time:\s*([0-9]+(?:\.[0-9]+)?)s\s*$", re.M),
    "transcribe_time": re.compile(r"^Transcribe time:\s*([0-9]+(?:\.[0-9]+)?)s\s*$", re.M),
    "avg_cpu": re.compile(r"^Avg CPU% during transcribe:\s*([0-9]+(?:\.[0-9]+)?)%\s*$", re.M),
    "ram_load": re.compile(r"^RAM used by load:\s*([0-9]+(?:\.[0-9]+)?)\s*GiB\s*$", re.M),
    "ram_transcribe": re.compile(r"^RAM used by transcribe:\s*([0-9]+(?:\.[0-9]+)?)\s*GiB\s*$", re.M),
    "vram_peak": re.compile(r"^VRAM peak:\s*([0-9]+(?:\.[0-9]+)?)\s*$", re.M),
    "gpu_power_max": re.compile(r"^GPU power max:\s*([0-9]+(?:\.[0-9]+)?)\s*W\s*$", re.M),
    "gpu_mem_max": re.compile(r"^GPU memory max \(nvidia-smi\):\s*([0-9]+(?:\.[0-9]+)?)\s*MiB\s*$", re.M),
}

GPU_UTIL_PATTERN = re.compile(
    r"^GPU util avg/max:\s*([0-9]+(?:\.[0-9]+)?)%\s*/\s*([0-9]+(?:\.[0-9]+)?)%\s*$",
    re.M,
)


def find_local_script(script_name):
    cwd = Path.cwd()
    candidates = [cwd / script_name, cwd.parent / script_name]
    for p in candidates:
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"{script_name} not found in current or parent directory")


def extract_error_hints(text):
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if any(rx.search(line) for rx in ERROR_HINT_PATTERNS):
            lines.append(line)
    return lines[:8]


def parse_run_metrics(output_text):
    m = {}
    for key, pattern in RUN_METRIC_PATTERNS.items():
        matches = pattern.findall(output_text)
        if not matches:
            m[key] = None
            continue
        last = matches[-1]
        if key in ("device", "compute_type"):
            m[key] = str(last).strip()
        else:
            try:
                m[key] = float(last)
            except Exception:
                m[key] = None

    gpu_match = GPU_UTIL_PATTERN.findall(output_text)
    if gpu_match:
        avg, maxv = gpu_match[-1]
        m["gpu_util_avg"] = float(avg)
        m["gpu_util_max"] = float(maxv)
    else:
        m["gpu_util_avg"] = None
        m["gpu_util_max"] = None

    return m


def percentile(values, p):
    if not values:
        return None
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    k = (len(vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return vals[int(k)]
    return vals[f] + (vals[c] - vals[f]) * (k - f)


def median_or_none(values):
    return statistics.median(values) if values else None


def first_non_empty(items):
    for v in items:
        if v is not None and str(v).strip():
            return str(v).strip()
    return None


def check_cuda(python_exe, env):
    try:
        check_code = (
            "import json, torch\n"
            "d = {\n"
            "    'cuda_available': bool(torch.cuda.is_available()),\n"
            "    'cuda_version': torch.version.cuda,\n"
            "    'torch_version': getattr(torch, '__version__', None),\n"
            "    'arch_list': [],\n"
            "    'device_name': None,\n"
            "    'device_capability': None,\n"
            "    'arch_supported': None,\n"
            "}\n"
            "if d['cuda_available']:\n"
            "    d['arch_list'] = list(torch.cuda.get_arch_list())\n"
            "    cap = torch.cuda.get_device_capability(0)\n"
            "    d['device_capability'] = f\"sm_{cap[0]}{cap[1]}\"\n"
            "    d['device_name'] = torch.cuda.get_device_name(0)\n"
            "    d['arch_supported'] = d['device_capability'] in d['arch_list']\n"
            "print(json.dumps(d))\n"
        )
        check_cmd = [python_exe, "-c", check_code]
        p = subprocess.run(check_cmd, capture_output=True, text=True, env=env)
        stdout = (p.stdout or "").strip()
        stderr = (p.stderr or "").strip()
        if not stdout:
            return False, {"error": stderr or "no output"}
        try:
            info = json.loads(stdout)
            return bool(info.get("cuda_available")), info
        except Exception:
            return False, {"error": stdout}
    except Exception as e:
        return False, {"error": str(e)}


def format_run_list(values):
    return ", ".join(f"{v:.2f}" for v in values)


def generate_chart(python_exe, chart_script_path, summary_path, chart_output, repeats, env):
    title = f"Whisper Transcribe Time Comparison (repeats={repeats})"
    cmd = [
        python_exe,
        chart_script_path,
        "--input",
        str(summary_path),
        "--output",
        str(chart_output),
        "--title",
        title,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if p.returncode == 0:
        print(f"Wrote chart to {chart_output}")
        return True
    print("WARNING: chart generation failed.")
    stderr = (p.stderr or "").strip()
    stdout = (p.stdout or "").strip()
    if stderr:
        print(stderr)
    elif stdout:
        print(stdout)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default="king.mp3")
    ap.add_argument("--backend", choices=["hf", "faster", "both"], default="hf")
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--gpu_index", default=0, type=int)
    ap.add_argument(
        "--compute_type",
        default="float16",
        help="Compute type for faster-whisper backend (e.g. float16, int8_float16, int8).",
    )
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--output", default="performance_results.txt")
    ap.add_argument("--full_output_dir", default=None, help="Optional dir to save full raw outputs")
    ap.add_argument("--repeats", type=int, default=1, help="How many runs per model.")
    ap.add_argument(
        "--chart_output",
        default="transcribe_times.svg",
        help="Output SVG chart path generated automatically after benchmark.",
    )
    ap.add_argument(
        "--clear_proxy",
        action="store_true",
        help="Unset HTTP(S)/ALL proxy env vars for child processes (helps when local proxy settings break downloads).",
    )
    args = ap.parse_args()

    if args.repeats < 1:
        print("ERROR: --repeats must be >= 1")
        return

    selected_backends = ["hf", "faster"] if args.backend == "both" else [args.backend]
    script_paths = {}
    for backend in selected_backends:
        script_name = SCRIPT_BY_BACKEND[backend]
        try:
            script_paths[backend] = find_local_script(script_name)
        except FileNotFoundError as e:
            print(e)
            return

    try:
        chart_script_path = find_local_script("plot_transcribe_times.py")
    except FileNotFoundError as e:
        chart_script_path = None
        print(f"WARNING: {e}")

    child_env = os.environ.copy()
    if args.clear_proxy:
        for key in PROXY_ENV_KEYS:
            child_env.pop(key, None)

    bad_proxy_vars = []
    for key in PROXY_ENV_KEYS:
        val = (os.environ.get(key) or "").strip()
        if val and re.search(r"(127\.0\.0\.1|localhost):9\b", val, re.I):
            bad_proxy_vars.append(f"{key}={val}")
    if bad_proxy_vars and not args.clear_proxy:
        print("WARNING: Proxy environment appears misconfigured and may block model downloads:")
        for item in bad_proxy_vars:
            print(f" - {item}")
        print("Tip: rerun with --clear_proxy or unset these variables in your shell.\n")

    print(f"Using Python interpreter: {args.python}")
    ok_cuda, cuda_info = check_cuda(args.python, child_env)
    if args.device.lower() == "gpu":
        need_torch_cuda_check = "hf" in selected_backends
        if not ok_cuda:
            if need_torch_cuda_check:
                print("ERROR: GPU requested but CUDA is not available to the selected Python.\n")
                print("Details:")
                if isinstance(cuda_info, dict):
                    if cuda_info.get("error"):
                        print(cuda_info.get("error"))
                    else:
                        print(f" - Torch: {cuda_info.get('torch_version')}")
                        print(f" - CUDA build: {cuda_info.get('cuda_version')}")
                        print(f" - Device detected: {cuda_info.get('device_name')}")
                else:
                    print(cuda_info)
                print(
                    "\nCommon fixes:\n"
                    " - Install a CUDA-enabled PyTorch wheel that matches your GPU and driver (see https://pytorch.org).\n"
                    " - Ensure correct GPU drivers are installed.\n"
                    " - Activate the same virtualenv you installed the CUDA PyTorch into.\n"
                    " - If you want to continue on CPU, run with --device cpu."
                )
                return
            print("WARNING: torch CUDA preflight is unavailable; continuing because backend is faster-whisper only.")
        else:
            if isinstance(cuda_info, dict) and cuda_info.get("arch_supported") is False:
                if need_torch_cuda_check:
                    print("ERROR: CUDA is available, but this PyTorch build does not support your GPU architecture.\n")
                    print("Details:")
                    print(f" - Torch: {cuda_info.get('torch_version')}")
                    print(f" - CUDA build: {cuda_info.get('cuda_version')}")
                    print(f" - GPU: {cuda_info.get('device_name')} ({cuda_info.get('device_capability')})")
                    print(f" - Supported architectures: {' '.join(cuda_info.get('arch_list') or [])}")
                    print(
                        "\nFix: install a newer CUDA-enabled PyTorch build that includes your GPU arch.\n"
                        "For RTX 50-series cards, use the latest build from the PyTorch selector (often CUDA 12.8+ or newer/nightly)."
                    )
                    return
                print("WARNING: torch reports unsupported GPU arch, but continuing for faster-whisper backend.")
            print(
                "CUDA available "
                f"(torch {cuda_info.get('torch_version')}, CUDA {cuda_info.get('cuda_version')}, "
                f"device {cuda_info.get('device_name')} {cuda_info.get('device_capability')}). Proceeding on GPU."
            )

    out_path = Path(args.output)
    if args.full_output_dir:
        full_dir = Path(args.full_output_dir)
        full_dir.mkdir(parents=True, exist_ok=True)
    else:
        full_dir = None

    with out_path.open("w", encoding="utf-8") as f:
        f.write("Performance summary\n")
        f.write("===================\n\n")
        for backend in selected_backends:
            models = HF_MODELS if backend == "hf" else FASTER_MODELS
            script_path = script_paths[backend]

            f.write(f"Backend: {backend}\n")
            f.write("-" * (9 + len(backend)) + "\n\n")

            for model in models:
                print(f"Starting [{backend}] model: {model} (repeats={args.repeats})")
                f.write(f"Model: {model}\n")
                f.write("-" * (7 + len(model)) + "\n")
                f.write(f"Backend: {backend}\n")
                f.write(f"Repeats: {args.repeats}\n")

                run_results = []
                all_error_hints = []
                model_log_paths = []

                for run_idx in range(1, args.repeats + 1):
                    print(f"  Run {run_idx}/{args.repeats}")

                    if full_dir:
                        model_log_name = f"{backend}_{model.replace('/', '_')}_run{run_idx:02d}.log"
                        model_log_path = full_dir / model_log_name
                        model_log_f = model_log_path.open("w", encoding="utf-8")
                        model_log_paths.append(model_log_path)
                    else:
                        model_log_f = None

                    cmd = [
                        args.python,
                        script_path,
                        args.audio,
                        "--device",
                        args.device,
                        "--gpu_index",
                        str(args.gpu_index),
                        "--model",
                        model,
                    ]
                    if backend == "faster":
                        cmd.extend(["--compute_type", args.compute_type])

                    proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        env=child_env,
                    )
                    collected = []
                    try:
                        for line in proc.stdout:
                            print(line, end="")
                            if model_log_f:
                                model_log_f.write(line)
                                model_log_f.flush()
                            collected.append(line)
                    except Exception:
                        pass
                    proc.wait()
                    if model_log_f:
                        model_log_f.close()

                    out_text = "".join(collected)
                    all_error_hints.extend(extract_error_hints(out_text))
                    run_results.append(
                        {
                            "run_idx": run_idx,
                            "exit_code": int(proc.returncode),
                            "metrics": parse_run_metrics(out_text),
                        }
                    )

                aggregate_exit = next((r["exit_code"] for r in run_results if r["exit_code"] != 0), 0)
                successful = [
                    r for r in run_results if r["exit_code"] == 0 and r["metrics"].get("transcribe_time") is not None
                ]
                f.write(f"Successful runs: {len(successful)}/{args.repeats}\n")
                f.write(f"Exit code: {aggregate_exit}\n")

                if successful:
                    transcribe_runs = [r["metrics"]["transcribe_time"] for r in successful if r["metrics"]["transcribe_time"] is not None]
                    transcribe_median = median_or_none(transcribe_runs)
                    transcribe_p95 = percentile(transcribe_runs, 95)
                    transcribe_min = min(transcribe_runs)
                    transcribe_max = max(transcribe_runs)

                    f.write(f"Transcribe runs (s): {format_run_list(transcribe_runs)}\n")
                    f.write(f"Transcribe time: {transcribe_median:.2f}s\n")
                    if transcribe_p95 is not None:
                        f.write(f"Transcribe p95: {transcribe_p95:.2f}s\n")
                    f.write(f"Transcribe min/max: {transcribe_min:.2f}s / {transcribe_max:.2f}s\n")

                    device = first_non_empty(r["metrics"].get("device") for r in successful)
                    compute_type = first_non_empty(r["metrics"].get("compute_type") for r in successful)
                    if device:
                        f.write(f"Device: {device}\n")
                    if compute_type:
                        f.write(f"Compute type: {compute_type}\n")

                    load_med = median_or_none([r["metrics"]["load_time"] for r in successful if r["metrics"]["load_time"] is not None])
                    avg_cpu_med = median_or_none([r["metrics"]["avg_cpu"] for r in successful if r["metrics"]["avg_cpu"] is not None])
                    ram_load_med = median_or_none([r["metrics"]["ram_load"] for r in successful if r["metrics"]["ram_load"] is not None])
                    ram_tx_med = median_or_none([r["metrics"]["ram_transcribe"] for r in successful if r["metrics"]["ram_transcribe"] is not None])
                    vram_med = median_or_none([r["metrics"]["vram_peak"] for r in successful if r["metrics"]["vram_peak"] is not None])
                    gpu_util_avg_med = median_or_none([r["metrics"]["gpu_util_avg"] for r in successful if r["metrics"]["gpu_util_avg"] is not None])
                    gpu_util_max_med = median_or_none([r["metrics"]["gpu_util_max"] for r in successful if r["metrics"]["gpu_util_max"] is not None])
                    gpu_power_med = median_or_none([r["metrics"]["gpu_power_max"] for r in successful if r["metrics"]["gpu_power_max"] is not None])
                    gpu_mem_med = median_or_none([r["metrics"]["gpu_mem_max"] for r in successful if r["metrics"]["gpu_mem_max"] is not None])

                    if load_med is not None:
                        f.write(f"Load time: {load_med:.2f}s\n")
                    if avg_cpu_med is not None:
                        f.write(f"Avg CPU% during transcribe: {avg_cpu_med:.1f}%\n")
                    if ram_load_med is not None:
                        f.write(f"RAM used by load: {ram_load_med:.3f} GiB\n")
                    if ram_tx_med is not None:
                        f.write(f"RAM used by transcribe: {ram_tx_med:.3f} GiB\n")
                    if vram_med is not None:
                        f.write(f"VRAM peak: {vram_med}\n")
                    if gpu_util_avg_med is not None and gpu_util_max_med is not None:
                        f.write(f"GPU util avg/max: {gpu_util_avg_med:.1f}% / {gpu_util_max_med:.1f}%\n")
                    if gpu_power_med is not None:
                        f.write(f"GPU power max: {gpu_power_med:.1f} W\n")
                    if gpu_mem_med is not None:
                        f.write(f"GPU memory max (nvidia-smi): {gpu_mem_med:.0f} MiB\n")
                else:
                    f.write("No performance lines detected.\n")
                    unique_hints = []
                    for hint in all_error_hints:
                        if hint not in unique_hints:
                            unique_hints.append(hint)
                    if unique_hints:
                        f.write("Error hints:\n")
                        for hint in unique_hints[:12]:
                            f.write(f"{hint}\n")

                if model_log_paths:
                    if len(model_log_paths) == 1:
                        f.write(f"Full output: {model_log_paths[0]}\n")
                    else:
                        for path in model_log_paths:
                            f.write(f"Full output: {path}\n")

                f.write("\n")
                f.flush()
            f.write("\n")

    print(f"Wrote performance summary to {out_path}")

    if chart_script_path:
        generate_chart(
            python_exe=args.python,
            chart_script_path=chart_script_path,
            summary_path=out_path,
            chart_output=args.chart_output,
            repeats=args.repeats,
            env=child_env,
        )


if __name__ == "__main__":
    main()
