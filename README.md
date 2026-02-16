# Whisper-Model-Tester

CPU 

pip install transformers psutil
pip install torch torchvision torchaudio

GPU 

pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers psutil

Install ffmpeg

winget install Gyan.FFmpeg

Run CPU 

python resultsWhisper.py king.mp3 --device cpu --model openai/whisper-small

Run GPU 

python resultsWhisper.py king.mp3 --device gpu --gpu_index 0 --model openai/whisper-small
