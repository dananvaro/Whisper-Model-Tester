# HuggingFace pipeline, time utilities, psutil system/process info and gc garbage collector
from transformers import pipeline
import time
import psutil
import gc

whisperModels = [
    "NBAiLab/nb-whisper-tiny"

]

audioFile = "king.mp3"

for model in whisperModels:


    # RAM used before test
    gc.collect()
    ramStart = psutil.virtual_memory().used /  (1024*3)

    # Loading model 
    # Times start for loading model
    loadStart = time.time()
    asr = pipeline("automatic-speech-recognition", model)
    loadEnd = time.time()
    loadTime = loadEnd - loadStart


print(loadTime)




