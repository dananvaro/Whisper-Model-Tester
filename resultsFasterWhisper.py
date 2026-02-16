                                                                                                            import argparse
import os
import subprocess
import threading
import time

import psutil

try:
    from faster_whisper import WhisperModel

    FASTER_OK = True
except Exception:
    FASTER_OK = False

try:
    import torch

    TORCH_OK = True
except Exception:
    TORCH_OK = False


def convertToGiB(b):
    return b / (1024**3)


class GPUSampler:
    """Poll nvidia-smi while inference runs to estimate GPU activity."""

    def __init__(self, gpu_index=0, interval_sec=0.2):
        self.gpu_index = int(gpu_index)
        self.interval_sec = float(interval_sec)
        self._stop = threading.Event()
        self._thread = None
        self.util_samples = []
        self.mem_samples = []
        self.power_samples = []
        self.error = None

    def _poll_once(self):
        cmd = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,power.draw",
            "--format=csv,noheader,nounits",
            "-i",
            str(self.gpu_index),
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
        if not out:
            return
        # Expected CSV line: "<util>, <mem_mib>, <power_w>"
        parts = [p.strip() for p in out.split(",")]
        if len(parts) < 3:
            return
        self.util_samples.append(float(parts[0]))
        self.mem_samples.append(float(parts[1]))
        self.power_samples.append(float(parts[2]))

    def _run(self):
        while not self._stop.is_set():
            try:
                self._poll_once()
            except Exception as e:
                self.error = str(e)
                break
            self._stop.wait(self.interval_sec)

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def summary(self):
        if not self.util_samples:
            return {
                "gpuUtilAvg": None,
                "gpuUtilMax": None,
                "gpuMemMaxMiB": None,
                "gpuPowerMaxW": None,
                "gpuSamples": 0,
                "gpuSamplerError": self.error,
            }
        n = len(self.util_samples)
        return {
            "gpuUtilAvg": sum(self.util_samples) / n,
            "gpuUtilMax": max(self.util_samples),
            "gpuMemMaxMiB": max(self.mem_samples) if self.mem_samples else None,
            "gpuPowerMaxW": max(self.power_samples) if self.power_samples else None,
            "gpuSamples": n,
            "gpuSamplerError": self.error,
        }


def transcribe(model, audio, device, gpu_index, compute_type, language, beam_size):
    sysRes = psutil.Process(os.getpid())
    baselineRam = sysRes.memory_info().rss

    load0 = time.time()
    asr = WhisperModel(
        model_size_or_path=model,
        device=device,
        device_index=gpu_index,
        compute_type=compute_type,
    )
    loadTime = time.time() - load0
    ramAfterLoad = sysRes.memory_info().rss

    transcribeTime0 = time.time()
    cpu = time.process_time()
    gpu_sampler = None
    if device == "cuda":
        gpu_sampler = GPUSampler(gpu_index=gpu_index)
        gpu_sampler.start()

    try:
        segments, _info = asr.transcribe(
            audio,
            beam_size=beam_size,
            task="transcribe",
            language=language,
        )
        text = "".join(seg.text for seg in segments).strip()
    finally:
        if gpu_sampler is not None:
            gpu_sampler.stop()

    runTime = time.time() - transcribeTime0
    cpuTime = time.process_time() - cpu

    avgCPUUsage = (cpuTime / runTime) * 100.0 if runTime > 0 else 0.0
    ramUsed = sysRes.memory_info().rss

    gpuStats = gpu_sampler.summary() if gpu_sampler is not None else {
        "gpuUtilAvg": None,
        "gpuUtilMax": None,
        "gpuMemMaxMiB": None,
        "gpuPowerMaxW": None,
        "gpuSamples": 0,
        "gpuSamplerError": None,
    }
    vramPeak = None
    if gpuStats["gpuMemMaxMiB"] is not None:
        vramPeak = gpuStats["gpuMemMaxMiB"] / 1024.0

    return {
        "text": text,
        "loadTime": loadTime,
        "runTime": runTime,
        "avgCPU": avgCPUUsage,
        "ramLoad": convertToGiB(ramAfterLoad - baselineRam),
        "ramAfterTranscribe": convertToGiB(ramUsed - ramAfterLoad),
        "vramPeak": vramPeak,
        "gpuUtilAvg": gpuStats["gpuUtilAvg"],
        "gpuUtilMax": gpuStats["gpuUtilMax"],
        "gpuMemMaxMiB": gpuStats["gpuMemMaxMiB"],
        "gpuPowerMaxW": gpuStats["gpuPowerMaxW"],
        "gpuSamples": gpuStats["gpuSamples"],
        "gpuSamplerError": gpuStats["gpuSamplerError"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Faster-Whisper benchmark runner")
    parser.add_argument("audio_file", type=str, help="Path to audio file (wav/mp3/etc)")
    parser.add_argument("--model", type=str, default="large-v3", help="faster-whisper model id")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu", help="Run on CPU or GPU")
    parser.add_argument("--gpu_index", type=int, default=0, help="CUDA GPU index (0 = first GPU)")
    parser.add_argument(
        "--compute_type",
        type=str,
        default="float16",
        help="faster-whisper compute type (e.g. float16, int8_float16, int8)",
    )
    parser.add_argument("--language", type=str, default="no", help="Language code")
    parser.add_argument("--beam_size", type=int, default=1, help="Beam size")
    args = parser.parse_args()

    if not FASTER_OK:
        print("ERROR: faster-whisper is not installed.")
        print("Install with: python -m pip install faster-whisper")
        raise SystemExit(2)

    if args.device == "cpu":
        device_id = "cpu"
        if args.compute_type == "float16":
            compute_type = "int8"
        else:
            compute_type = args.compute_type
    else:
        if TORCH_OK and not torch.cuda.is_available():
            print("GPU requested, but CUDA is not available. Falling back to CPU.")
            device_id = "cpu"
            compute_type = "int8"
        else:
            device_id = "cuda"
            compute_type = args.compute_type

    out = transcribe(
        model=args.model,
        audio=args.audio_file,
        device=device_id,
        gpu_index=args.gpu_index,
        compute_type=compute_type,
        language=args.language,
        beam_size=args.beam_size,
    )

    print(f"\nModel: {args.model}")
    print(f"Device: {'CPU' if device_id == 'cpu' else f'GPU:{args.gpu_index}'}")
    print(f"Compute type: {compute_type}")
    print(f"Load time: {out['loadTime']:.2f}s")
    print(f"Transcribe time: {out['runTime']:.2f}s")
    print(f"Avg CPU% during transcribe: {out['avgCPU']:.1f}%")
    print(f"RAM used by load: {out['ramLoad']:.3f} GiB")
    print(f"RAM used by transcribe: {out['ramAfterTranscribe']:.3f} GiB")
    print(f"VRAM peak: {out['vramPeak'] if out['vramPeak'] is not None else 'N/A'}")
    if out["gpuSamples"] > 0:
        print(f"GPU util avg/max: {out['gpuUtilAvg']:.1f}% / {out['gpuUtilMax']:.1f}%")
        print(f"GPU power max: {out['gpuPowerMaxW']:.1f} W")
        print(f"GPU memory max (nvidia-smi): {out['gpuMemMaxMiB']:.0f} MiB")
    elif out["gpuSamplerError"]:
        print(f"GPU telemetry: unavailable ({out['gpuSamplerError']})")
    print(f"Text: {out['text']}")
