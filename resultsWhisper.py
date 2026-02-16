# HuggingFace pipeline, time utilities, psutil system/process info and gc garbage collector
import time, os, gc, warnings, argparse
import psutil
from transformers import pipeline


try:
    import torch
    TORCH_OK =True
except Exception:
    TORCH_OK =False

def convertToGiB(b):
    return (b/(1024**3))


def transcribe(model, audio, device):

    sysRes = psutil.Process(os.getpid())

    baselineRam = sysRes.memory_info().rss

    if TORCH_OK and device != -1 and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    # Loading model
    load0 = time.time()
    asr = pipeline("automatic-speech-recognition", model=model, device=device)
    loadTime = time.time() - load0
    ramAfterLoad = sysRes.memory_info().rss

    # Transcribe
    transcribeTime0 = time.time()
    cpu= time.process_time()

    results = asr(audio,return_timestamps=True,generate_kwargs={"num_beams": 1, "task": "transcribe", "language": "no"},)
    runTime = time.time() - transcribeTime0
    
    cpuTime = time.process_time() - cpu

    # CPU Usage in % 

    avgCPUUSage = (cpuTime/runTime) * 100.0

   

    ramUsed= sysRes.memory_info().rss

    peakVRAM = None
    if TORCH_OK and device != -1 and torch.cuda.is_available():
        peakVRAM = convertToGiB(torch.cuda.max_memory_allocated())
        torch.cuda.empty_cache()

    return {
        "text": results["text"],
        "loadTime": loadTime,
        "runTime" : runTime,
        "avgCPU" : avgCPUUSage,
        "ramLoad": convertToGiB(ramAfterLoad - baselineRam),
        "ramAfterTranscribe" : convertToGiB(ramUsed-ramAfterLoad),
        "vramPeak": peakVRAM

    }

if __name__ == "__main__":

  
    # FIX: your code below still has torch.cude -> torch.cuda.
    # Easiest: change that line inside transcribe() to torch.cuda (or remove it).
    # For now, this main will run CPU by default and can run GPU if available.

    parser = argparse.ArgumentParser(description="Quick test runner")
    parser.add_argument("audio_file", type=str, help="Path to audio file (wav/mp3/etc)")
    parser.add_argument("--model", type=str, default="NbAiLab/nb-whisper-small", help="HF model id")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu", help="Run on CPU or GPU")
    parser.add_argument("--gpu_index", type=int, default=0, help="CUDA GPU index (0 = first GPU)")
    args = parser.parse_args()

    # Pick device id for transformers pipeline
    if args.device == "cpu":
        device_id = -1
    else:
        if not (TORCH_OK and torch.cuda.is_available()):
            print("GPU requested, but CUDA is not available. Falling back to CPU.")
            device_id = -1
        else:
            device_id = args.gpu_index

    # Run
    out = transcribe(args.model, args.audio_file, device_id)

    # Print results
    print(f"\nModel: {args.model}")
    print(f"Device: {'CPU' if device_id == -1 else f'GPU:{device_id}'}")
    print(f"Load time: {out['loadTime']:.2f}s")
    print(f"Transcribe time: {out['runTime']:.2f}s")
    print(f"Avg CPU% during transcribe: {out['avgCPU']:.1f}%")
    print(f"RAM used by load: {out['ramLoad']:.3f} GiB")
    print(f"RAM used by transcribe: {out['ramAfterTranscribe']:.3f} GiB")
    print(f"VRAM peak: {out['vramPeak'] if out['vramPeak'] is not None else 'N/A'}")
    print(f"Text: {out['text']}")