# HuggingFace pipeline, time utilities, psutil system/process info and gc garbage collector
from transformers import pipeline
import time
import psutil
import gc
import os
import warnings
from transformers.utils import logging

logging.set_verbosity_error()

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] ="1"

warnings.filterwarnings("ignore", message="Using `chunk_length_s` is very experimental*")
warnings.filterwarnings("ignore", message="You are sending unauthenticated requests to the HF Hub*")
warnings.filterwarnings("ignore", message="`huggingface_hub` cache-system uses symlinks*")
warnings.filterwarnings("ignore", category=FutureWarning)




whisperModels = [
    "NbAiLab/nb-whisper-tiny",
    "NbAiLab/nb-whisper-base",
    "NbAiLab/nb-whisper-small",
    "NbAiLab/nb-whisper-medium",
    "NbAiLab/nb-whisper-large"

]

audioFile = "king.mp3"

for model in whisperModels:


    # RAM used before 
    gc.collect()
    ramStart = psutil.virtual_memory().used /  (1024*3)

    # Loading model 
    # Times start for loading model
    loadStart = time.time()
    asr = pipeline("automatic-speech-recognition", model)
    loadEnd = time.time()
    loadTime = loadEnd - loadStart

    # RAM used after 
    ramEnd = psutil.virtual_memory().used /(1024*3)
    ramUsed = ramEnd-ramStart

    # Transcribe 
    transcribeStart = time.time()
    transcribe = asr("king.mp3", chunk_length_s=28, return_timestamps=True, generate_kwargs={'num_beams': 5, 'task': 'transcribe', 'language': 'no'})
    transcribeEnd = time.time()
    transcribeSeconds = transcribeEnd - transcribeStart

    print(f"Model: {model} \\ Transcribe seconds: {transcribeSeconds} \\ Output: {transcribe['text']}")




