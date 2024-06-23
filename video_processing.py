# Standard library imports
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from threading import Semaphore, Lock

# Third-party imports for concurrent and multiprocessing tasks
import concurrent.futures
import multiprocessing
import threading
from multiprocessing import Process, Manager

# NLTK and Spacy for natural language processing
from nltk import download
import spacy

# Libraries for video and audio processing
from moviepy.editor import VideoFileClip
import speech_recognition as sr

# Libraries for web content and translation
from pytube import YouTube
from translate import Translator

# Library for emotion analysis
from nrclex import NRCLex

# TextBlob for sentiment analysis
from textblob import TextBlob

# Download necessary NLTK data and load Spacy model
download('punkt')
nlp = spacy.load('en_core_web_sm')

#########################################################

# Here's the core function that will download a YT video from a URL and save
def youtube_downloader(url, output_path):
    yt = YouTube(url)
    stream = yt.streams.get_highest_resolution()
    stream.download(output_path=output_path)
    video_title = yt.title

#########################################################

def download_video(url, output_path, semaphore):
    with semaphore:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        print(f"{now} - Starting download: {url[-20:]}")
        try:
            youtube_downloader(url, output_path)
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            print(f"{now} - Finished download: {url[-20:]}")
        except Exception as e:
            print(f"{now} - Failed download: {url[-20:]}")

def download_videos_multiprocessing(urls, output_path, semaphore):
    start = datetime.now()
    
    processes = []
    for url in urls:
        process = Process(target=download_video, args=(url, output_path, semaphore))
        processes.append(process)
        process.start()
    for process in processes:
        process.join()
    
    end = datetime.now()
    
    print('.' * 65)
    print("* ", f'download videos with multiprocessing: {(end-start).total_seconds()} second(s)')
    print('.' * 65)


def download_videos_threading(urls, output_path, semaphore):
    start = time.perf_counter()
    threads = []
    for url in urls:
        thread = threading.Thread(target=download_video, args=(url, output_path, semaphore))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
    end = time.perf_counter()
    
    print('.' * 65)
    print("* ", f'download videos with threading: {end-start} second(s)')
    print('.' * 65)



# Setup logging
logging.basicConfig(filename='download_log.txt', level=logging.INFO, format='%(message)s')

def download_video_and_log(url, output_path, semaphore, thread_lock):
    with semaphore:
        identifier = f'Thread {threading.get_ident()}'
        now = datetime.now().strftime("%H:%M, %d %b %Y")
        try:
            youtube_downloader(url, output_path)
            log_message = f'"Timestamp": {now}, "URL":"{url}", "Download":True'
            with thread_lock:
                logging.info(log_message)
            print(f"{now} - {identifier} - Finished download: {url[-20:]}")
        except Exception as e:
            log_message = f'"Timestamp": {now}, "URL":"{url}", "Download":False, "Error":"{str(e)}"'
            with thread_lock:
                logging.error(log_message)
            print(f"{now} - {identifier} - Failed download: {url[-20:]} - {str(e)}")


def run_download_video_and_log(urls, output_path, semaphore, thread_lock):
    start = time.perf_counter()
    threads = []
    for url in urls:
        thread = threading.Thread(target=download_video_and_log, args=(url, output_path, semaphore, thread_lock))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
    end = time.perf_counter()
        
    print('.' * 65)
    print("* ", f'download videos with threading: {end-start} second(s)')
    print('.' * 65)


#########################################################

def extract_audio(video_path, audio_output_path):
    file_name = os.path.splitext(os.path.basename(video_path))[0] + '.wav'
    audio_path = os.path.join(audio_output_path, file_name)
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_path, codec='pcm_s16le', verbose=False)
    audio.close()
    video.close()
    print(f"Extracted audio to {audio_path}")


def run_extract_audio(video_output_path, audio_output_path):
    start = time.perf_counter()
    if not os.path.exists(audio_output_path):
        os.makedirs(audio_output_path)
    videos = [os.path.join(video_output_path, f) for f in os.listdir(video_output_path) if f.endswith(('.mp4'))]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(extract_audio, video, audio_output_path) for video in videos]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f'An exception occurred: {exc}')
    end = time.perf_counter()
    print('.' * 65)
    print("* ", f'Extracting audio from videos took {end-start:.2f} second(s)')
    print('.' * 65)


#########################################################


def transcribe_audio(data):
    audio_path, text_output_path = data
    recognizer = sr.Recognizer()
    file_name = os.path.splitext(os.path.basename(audio_path))[0] + '.txt'
    text_path = os.path.join(text_output_path, file_name)
    try:
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        with open(text_path, 'w') as f:
            f.write(text)
        print(f"Transcribed '{audio_path}' to '{text_path}'")
    except Exception as e:
        print(f"Failed to transcribe '{audio_path}': {str(e)}")


def run_transcribe_audio(audio_output_path, text_output_path):
    start = time.perf_counter()
    if not os.path.exists(text_output_path):
        os.makedirs(text_output_path)
    audio_files = [os.path.join(audio_output_path, f) for f in os.listdir(audio_output_path) if f.endswith('.wav')]
    data = [(audio_file, text_output_path) for audio_file in audio_files]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(transcribe_audio, data)

    end = time.perf_counter()
    print('.' * 65)
    print("* ", f'Transcribing text from audio took {end-start:.2f} second(s)')
    print('.' * 65)


#########################################################

def analyze_sentiment(text_file, output_directory):
    with open(text_file, 'r', encoding='utf-8') as file:
        text = file.read()

    blob = TextBlob(text)
    json_data = {
        os.path.basename(text_file): {
            'sentiment': blob.sentiment
        }
    }

    json_file_name = os.path.splitext(os.path.basename(text_file))[0] + '.json'
    json_file_path = os.path.join(output_directory, json_file_name)
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(json_data, json_file, indent=4)


def run_analyze_sentiment(text_input_path, json_output_path):
    start = time.perf_counter()
    if not os.path.exists(json_output_path):
        os.makedirs(json_output_path)

    text_files = [os.path.join(text_input_path, f) for f in os.listdir(text_input_path) if f.endswith('.txt')]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(analyze_sentiment, text_files, [json_output_path] * len(text_files))

    end = time.perf_counter()
    print('.' * 65)
    print("* ", f'Transcribing text from audio took {end-start:.2f} second(s)')
    print('.' * 65)



#########################################################


def translate_file(file_data):
    file_path, input_directory, output_directory = file_data
    translator = Translator(to_lang="es")
    with open(os.path.join(input_directory, file_path), 'r', encoding='utf-8') as file:
        text = file.read()
    translation = translator.translate(text)
    output_file_path = os.path.join(output_directory, file_path)
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(translation)
    print(f"Translated '{file_path}' to Spanish and saved to '{output_file_path}'")

def run_translate_file(input_directory, output_directory):
    start = time.perf_counter()
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    text_files = [f for f in os.listdir(input_directory) if f.endswith('.txt')]
    file_data = [(file, input_directory, output_directory) for file in text_files]
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.map(translate_file, file_data)

    end = time.perf_counter()
    print('.' * 65)
    print("* ", f'translating text from english to spanish {end-start:.2f} second(s)')
    print('.' * 65)


#########################################################

def analyze_emotion(text_file, output_directory):
    try:
        with open(text_file, 'r', encoding='utf-8') as file:
            text = file.read()
    except Exception as e:
        print(f"Error reading {text_file}: {e}")
        return
    doc = nlp(text)
    full_text = ' '.join([sent.text for sent in doc.sents])
    emotion = NRCLex(full_text)
    emotion_data = emotion.affect_frequencies
    json_data = {
        'Detected Emotions and Frequencies': emotion_data
    }
    file_name = os.path.splitext(os.path.basename(text_file))[0] + '_emotions.json'
    json_file_path = os.path.join(output_directory, file_name)
    try:
        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(json_data, json_file, indent=4)
        print(f"Processed {text_file}: Emotions saved to {json_file_path}")
    except Exception as e:
        print(f"Error writing {json_file_path}: {e}")

def run_analyze_emotion(input_directory, output_directory):
    start = time.perf_counter()
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    text_files = [os.path.join(input_directory, f) for f in os.listdir(input_directory) if f.endswith('.txt')]
    for f in text_files:
        if not os.path.isfile(f):
            print(f"File not found: {f}")
    tasks = [(f, output_directory) for f in text_files]
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.starmap(analyze_emotion, tasks)
    end = time.perf_counter()
    print('.' * 65)
    print("* ", f'Emotion analysis process completed in {end-start:.2f} second(s)')
    print('.' * 65)
