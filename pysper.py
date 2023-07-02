import os
import streamlit as st
import whisper
from pyannote.audio import Pipeline
from utils import *
from tqdm import tqdm
import psutil
import time
import datetime
import sys
import torch
import gc

st.title("Pysper")
st.text("Upload File")

adjust_cpu_usage()
psutil.cpu_percent(interval=1, percpu=False)
device = torch.device("cuda:0")

audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a", "mp4", "avi"])

if audio_file is not None:
    asr_model = whisper.load_model("medium")
    audio_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token="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").to(device)
    asr_model = whisper.load_model("medium").to(device)
    asr_model.to(torch.device("cuda:0"))
    audio_pipeline.to(torch.device("cuda:0"))
    st.text("Model Loaded")

if st.sidebar.button("Transcribe Audio"):
    if audio_file is not None:
        with open(audio_file.name,"wb") as f:
            f.write(audio_file.getbuffer())
        audiofile_name = audio_file.name
        filetype = ["m4a", "mp3", "mp4", "avi"]
        main = audio_file.name.split(".")[0]
        main = f"{main}.wav"
        for element in filetype:
            if element in audiofile_name:
                if os.path.exists(main):
                    convert_audio_to_wav_2(audiofile_name)
                else:
                    convert_audio_to_wav_1(audiofile_name)
        if os.path.exists(audiofile_name):
            os.remove(audiofile_name)
        st.sidebar.success("Transcribe Audio")
        asr_transcription = asr_model.transcribe(main, verbose=False, language="en")
        for result in tqdm(range(1)):
            diarization = audio_pipeline(main)
        diarized_text = diarize_and_merge_text(asr_transcription, diarization)
        st.sidebar.success("Transcription Completed")
        for seg, speaker, sentence in tqdm(diarized_text):
            line = f'{seg.start:.2f} / {seg.end:.2f} / {speaker} / {sentence}\n'
            str(line).encode(encoding="utf8", errors="xmlcharrefreplace")
            st.markdown(line)
        if os.path.exists(main):
            os.remove(main)
        gc.collect()
        torch.cuda.empty_cache()
    else:
        st.sidebar.error("Please upload an audio file")

st.sidebar.header("Play Audio File")
st.sidebar.audio(audio_file)
