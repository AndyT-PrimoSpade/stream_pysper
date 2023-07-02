from pyannote.core import Segment, Annotation, Timeline
import math
from tqdm import tqdm
from datetime import datetime, timedelta
import os
from pydub import AudioSegment
import psutil
import time
import ffmpeg
import subprocess
import io
import ctypes
import tkinter as tk
import gc

def has_file():
    while True:
        audio_filename = input("Audio File Name: ")
        audiofile_name = f"../input/{audio_filename}"
        if not os.path.exists(audiofile_name):
            print("Invalid file name, please enter again")
            continue
        return audiofile_name

PUNCTUATION_SENTENCE_END = ['.', '?', '!', ' ']


def get_text_with_timestamp(asr_result):
    timestamp_texts = []
    for item in asr_result['segments']:
        start_time = item['start']
        end_time = item['end']
        text = item['text']
        timestamp_texts.append((Segment(start_time, end_time), text))
    return timestamp_texts


def add_speaker_info_to_text(timestamp_texts, diarization_result):
    spk_text = []
    for seg, text in timestamp_texts:
        print(seg)
        speaker = diarization_result.crop(seg).argmax()
        spk_text.append((seg, speaker, text))
        print(speaker)
    return spk_text


def merge_cache(text_cache, speaker):
    sentence = ''.join([item[-1] for item in text_cache])
    start_time = text_cache[0][0].start
    end_time = text_cache[-1][0].end
    return Segment(start_time, end_time), speaker, sentence


def merge_sentence(spk_text):
    merged_spk_text = []
    previous_speaker = None
    text_cache = []
    for seg, speaker, text in spk_text:
        print(f"{previous_speaker} - {speaker}")      
        if speaker is None:
            if previous_speaker is not None:    
                speaker = previous_speaker
        if speaker != previous_speaker and previous_speaker is not None and len(text_cache) > 0:
            merged_spk_text.append(merge_cache(text_cache, previous_speaker))
            text_cache = [(seg, speaker, text)]
            previous_speaker = speaker
        elif text[-1] in PUNCTUATION_SENTENCE_END:
            text_cache.append((seg, speaker, text))
            merged_spk_text.append(merge_cache(text_cache, speaker))
            text_cache = []
            previous_speaker = speaker
        else:
            text_cache.append((seg, speaker, text))
            previous_speaker = speaker
    if len(text_cache) > 0:
        merged_spk_text.append(merge_cache(text_cache, previous_speaker))
    return merged_spk_text


def diarize_and_merge_text(asr_result, diarization_result):
    timestamp_texts = get_text_with_timestamp(asr_result)
    spk_text = add_speaker_info_to_text(timestamp_texts, diarization_result)
    res_processed = merge_sentence(spk_text)
    return res_processed


def write_results_to_txt_file(final_result, file_name):
    if os.path.exists(file_name):
        os.remove(file_name)
    with open(file_name, 'w') as fp:
        for seg, speaker, sentence in tqdm(final_result):
            line = f'{seg.start:.2f} / {seg.end:.2f} / {speaker} / {sentence}\n'
            str(line).encode(encoding="utf8", errors="xmlcharrefreplace")
            fp.write(line)

def convert_txt_to_srt(input_file, output_file):
    with io.open(input_file, "r", encoding="utf-8") as input:
        lines = input.readlines()
    if os.path.exists(output_file):
        os.remove(output_file)
    else:
        pass
    srt_subtitles = []
    for i, line in tqdm(enumerate(lines)):
        start = line.split("/")[0].strip()
        start = float(start)
        start = int(start)
        start = timedelta(seconds = start)
        end = line.split("/")[1].strip()
        end = float(end)
        end = int(end)
        end = timedelta(seconds = end)
        speaker = line.split("/")[2].strip()
        main = line.split("/")[3].strip()
        content = f'{speaker} -- {main}'
        subtitles = f'{i+1}\n{start} --> {end}\n{content}\n\n'
        srt_subtitles.append(subtitles)
        with io.open(output_file, "a", encoding="utf-8") as output:
            output.write(subtitles)


def split_audio(fileName):
    audio_name = fileName
    audio_half = audio_name.split(".")[0]
    audio = AudioSegment.from_file(audio_name)

    duration = len(audio)

    half_point = int(duration / 2)
    extra_duration = duration % 2

    first_half = audio[:half_point]
    second_half = audio[half_point + extra_duration:]

    first_half.export(f"../audio/first_half.mp3")

    second_half.export(f"../audio/second_half.mp3")

def combine_txt_file(file1, file2, output):
    with io.open(file1, "r", encoding="utf-8") as f1:
        content1 = f1.read()
    with io.open(file2, "r", encoding="utf-8") as f2:
        content2 = f2.read()
    with io.open(output, "w", encoding="utf-8") as f3:
        f3.write(content1 + content2)

    if __name__ == "__main__":
        file1 = os.path.join(os.getcwd(), "file1.txt")
        file2 = os.path.join(os.getcwd(), "file2.txt")
        output = os.path.join(os.getcwd(), "output.txt")

def whisper_txt_combine(input_1, input_2):    
    with io.open("../output/whisper1.txt", 'w', encoding="utf-8") as text:
        text.write(input_1)
    with io.open("../tput/whisper2.txt", 'w', encoding="utf-8") as text:
        text.write(input_2)
    combine_txt_file("../output/whisper1.txt", "../output/whisper2.txt", "../output/whisper.txt")
    if os.path.exists("../output/whisper1.txt"):
        os.remove("../output/whisper1.txt")
    if os.path.exists("../output/whisper2.txt"):
        os.remove("../output/whisper2.txt")

def adjust_cpu_usage():
    cpu_limit = 20
    while True:
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > cpu_limit:
            time.sleep(0.1)
        else:
            break

def convert_m4a_to_wav(input_file, output_file):
    ffmpeg.input(input_file).output(output_file, format='wav').run()
    print("Conversion Completed!")

def convert_audio_to_wav(file_path):
    filenaming = file_path.split("/")[-1].split(".")[0]
    ffmpeg_command = ["ffmpeg","y", "-i", file_path, "-c:a", "pcm_s16le", "-ar", "44100", "-ac", "2", "-f", "wav", f"../convert/{filenaming}.wav"]
    subprocess.run(ffmpeg_command)

def clear_cmd():
    os.system("cls")

def min_console():
    console_handle = ctypes.windll.kernel32.GetConsoleWindow()
    ctypes.windll.user32.ShowWindow(console_handle, 0)

def clear_purge():
    gc.collect()
    gc.collect(generation=2)

# This is to print the result on the console
# for seg, spk, sent in final_result:
#     line = f'{seg.start:.2f} {seg.end:.2f} {spk} {sent}'
#     print(line)
