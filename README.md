# Whisper-Model-Tester

## Install

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install FFmpeg:

```powershell
winget install Gyan.FFmpeg
```

## GPU Notes (Important)

If you have a newer GPU (for example RTX 50-series), do **not** keep an old CUDA wheel such as `cu118`.
Reinstall torch from the PyTorch selector so it matches your GPU:

https://pytorch.org/get-started/locally/

Example pattern:

```powershell
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch torchvision torchaudio --index-url <URL_FROM_PYTORCH_SELECTOR>
```

Verify GPU + arch support:

```powershell
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('available', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); print('archs', torch.cuda.get_arch_list())"
```

## Proxy / Download Issues

If model downloads show repeated `Retrying in 1s` lines, check proxy env vars:

```powershell
Get-ChildItem Env: | Where-Object { $_.Name -match 'proxy|PROXY' }
```

If they point to a dead local proxy (for example `127.0.0.1:9`), clear them:

```powershell
Remove-Item Env:HTTP_PROXY,Env:HTTPS_PROXY,Env:ALL_PROXY,Env:http_proxy,Env:https_proxy,Env:all_proxy -ErrorAction SilentlyContinue
```

You can also run `run_all_models.py` with `--clear_proxy` to strip proxy vars for child processes.

## Run

CPU:

```powershell
python resultsWhisper.py king.mp3 --device cpu --model openai/whisper-small
```

GPU:

```powershell
python resultsWhisper.py king.mp3 --device gpu --gpu_index 0 --model openai/whisper-small
```

Faster-Whisper (single model):

```powershell
python resultsFasterWhisper.py king.mp3 --device gpu --gpu_index 0 --model large-v3 --compute_type float16
```

Batch all models (Hugging Face backend):

```powershell
python run_all_models.py --backend hf --device gpu --repeats 5 --full_output_dir raw_logs --clear_proxy
```

Batch all models (Faster-Whisper backend):

```powershell
python run_all_models.py --backend faster --device gpu --compute_type float16 --repeats 5 --full_output_dir raw_logs --clear_proxy
```

Compare both backends in one run:

```powershell
python run_all_models.py --backend both --device gpu --compute_type float16 --repeats 5 --full_output_dir raw_logs --clear_proxy
```

`run_all_models.py` now aggregates repeated runs per model and writes:
- `Transcribe time` = median of successful runs
- `Transcribe p95`
- `Transcribe runs (s)` list

It also auto-generates the chart after each batch run (default: `transcribe_times.svg`).

Manual chart generation (optional):

```powershell
python plot_transcribe_times.py --input performance_results.txt --output transcribe_times.svg
```
