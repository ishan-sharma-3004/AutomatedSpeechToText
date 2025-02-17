import os
import librosa
import torch
import numpy as np
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")
processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")
def transcribe_audio(audio):
    audio = np.array(audio)
    if np.std(audio) == 0 or np.isnan(audio).any():
        print("[WARNING] Audio segment has zero variance or contains NaN values. Skipping...")
        return ""
    audio = librosa.util.normalize(audio)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        generated_ids = model.generate(inputs["input_features"])
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return transcription[0]
def split_audio_on_silence(audio, sample_rate, silence_threshold=-40, min_speech_length=0.5):
    non_silent_intervals = librosa.effects.split(audio, top_db=abs(silence_threshold), frame_length=1024, hop_length=512)
    segments = [audio[start:end] for start, end in non_silent_intervals]
    return segments, non_silent_intervals
folder = "/Users/ishansharma/Downloads/Projectsubmission/processed_data/"
transcript_dict = {}
for file in os.listdir(folder):
    if file.endswith(".wav"):
        filename = os.path.join(folder, file)
        print(f"[INFO] Processing file: {filename}")
        try:
            audio, sample_rate = librosa.load(filename, sr=16000)
            if len(audio) == 0 or np.isnan(audio).any():
                print(f"[WARNING] Skipping {file} due to invalid or empty audio.")
                continue
            segments, intervals = split_audio_on_silence(audio, sample_rate)
            transcript_dict[file] = []
            for i, (segment, (start, end)) in enumerate(zip(segments, intervals)):
                segment_transcription = transcribe_audio(segment)
                transcript_dict[file].append({"segment": i + 1, "transcription": segment_transcription, "start_time": round(start / sample_rate, 2), "end_time": round(end / sample_rate, 2)
                })
        except Exception as e:
            print(f"[ERROR] Failed to process {file}: {e}")
for file, transcriptions in transcript_dict.items():
    print(f"\n[RESULT] Transcriptions for {file}:")
    for entry in transcriptions:
        print(f" Segment {entry['segment']} - {entry['transcription']}")
        print(f" Start Time: {entry['start_time']}s, End Time: {entry['end_time']}s")
