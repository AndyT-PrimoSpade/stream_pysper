import os
import streamlit as st
import whisper
from pyannote.audio import Pipeline
from stream_utils import *
from tqdm import tqdm
import psutil
import time
import datetime
import sys
import torch


st.title("Pysper")

start = datetime.datetime.now()
psutil.cpu_percent(interval=1, percpu=False)
device = torch.device("cuda:0")

audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a", "mp4", "avi", "mpeg"])

model = whisper.load_model("medium")
st.text("Model Loaded")

st.text(audio_file)

# if audio_file is not None:
#     asr_model = whisper.load_model("medium")
#     audio_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=None).to(device)
#     asr_model = whisper.load_model("medium").to(device)
#     asr_model.to(torch.device("cuda:0"))
#     audio_pipeline.to(torch.device("cuda:0"))
#     st.text("Model Loaded")

main = audio_file.name
if st.sidebar.button("Transcribe Audio"):
    if audio_file is not None:
        st.sidebar.success("Transcribe Audio")
        transcription = model.transcribe(audio_file.name)
        st.sidebar.cuss("Transcription Completed")
        st.markdown(transcription["text"])
        # st.sidebar.success("Transcribe Audio")
        # asr_transcription = asr_model.transcribe(main, verbose=False, language="en")
        # diarization = audio_pipeline(main)
        # diarized_text = diarize_and_merge_text(asr_transcription, diarization)
        # st.sidebar.success("Transcription Completed")
        # for seg, speaker, sentence in tqdm(final_result):
        #     line = f'{seg.start:.2f} / {seg.end:.2f} / {speaker} / {sentence}\n'
        #     str(line).encode(encoding="utf8", errors="xmlcharrefreplace")
        #     st.markdown(line)
    else:
        st.sidebar.error("Please upload an audio file")

st.sidebar.header("Play Audio File")
st.sidebar.audio(audio_file)