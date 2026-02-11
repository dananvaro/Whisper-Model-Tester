# HuggingFace pipeline, time utilities, psutil system/process info and gc garbage collector
from transformers import pipeline
import time
import psutil
import gc

whisperModels = [
    "NBAiLab/nb-whisper-tiny",
    "NBAiLab/nb-whisper-base",
    "NBAiLab/nb-whisper-small",
    "NBAiLab/nb-whisper-medium",
    "NBAiLab/nb-whisper-large"

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

    transcribe = asr("king.mp3", chunk_length_s=28, return_timestamps=True, generate_kwargs={'num_beams': 5, 'task': 'transcribe', 'language': 'no'})



print(transcribe)




