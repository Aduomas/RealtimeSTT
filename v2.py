IS_DEBUG = True
# IS_DEBUG = False
USE_STEREO_MIX = False
LOOPBACK_DEVICE_NAME = "stereomix"
LOOPBACK_DEVICE_HOST_API = 0

import os
import re
import sys
import threading
from threading import Event
import queue
import time
from collections import deque
from difflib import SequenceMatcher

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np

EXTENDED_LOGGING = False
sentence_end_marks = [".", "!", "?", "。"]

detection_speed = 0.5  # set detection speed between 0.1 and 2.0

if detection_speed < 0.1:
    detection_speed = 0.1
if detection_speed > 2.0:
    detection_speed = 2.0

last_detection_pause = 0
last_prob_complete = 0
last_suggested_pause = 0
last_pause = 0
end_of_sentence_detection_pause = 0.3
maybe_end_of_sentence_detection_pause = 0.5
unknown_sentence_detection_pause = 0.8
ellipsis_pause = 1.7
punctuation_pause = 0.5
exclamation_pause = 0.4
question_pause = 0.3

hard_break_even_on_background_noise = 3.0
hard_break_even_on_background_noise_min_texts = 3
hard_break_even_on_background_noise_min_chars = 15
hard_break_even_on_background_noise_min_similarity = 0.99

if __name__ == "__main__":

    if EXTENDED_LOGGING:
        import logging

        logging.basicConfig(level=logging.DEBUG)

    from rich.console import Console
    from rich.live import Live
    from rich.text import Text
    from rich.panel import Panel

    console = Console()
    console.print("System initializing, please wait")

    from RealtimeSTT import AudioToTextRecorder
    from colorama import Fore, Style
    import colorama

    import torch
    import torch.nn.functional as F
    from transformers import (
        DistilBertTokenizerFast,
        DistilBertForSequenceClassification,
    )

    # Load sentence completion classification model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("mps")
    model_dir = "KoljaB/SentenceFinishedClassification"
    max_length = 128

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    classification_model = DistilBertForSequenceClassification.from_pretrained(
        model_dir
    )
    classification_model.to(device)
    classification_model.eval()

    # Label mapping
    label_map = {0: "Incomplete", 1: "Complete"}

    # We now want probabilities, not just a label
    def get_completion_probability(sentence, model, tokenizer, device, max_length):
        """
        Return the probability that the sentence is complete.
        """
        inputs = tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1).squeeze().tolist()
        # probabilities is [prob_incomplete, prob_complete]
        # We want the probability of being complete
        prob_complete = probabilities[1]
        return prob_complete

    # We have anchor points for probability to detection mapping
    # (probability, rapid_sentence_end_detection)
    anchor_points = [(0.0, 1.0), (1.0, 0)]

    def interpolate_detection(prob):
        # Clamp probability between 0.0 and 1.0 just in case
        p = max(0.0, min(prob, 1.0))
        # If exactly at an anchor point
        for ap_p, ap_val in anchor_points:
            if abs(ap_p - p) < 1e-9:
                return ap_val

        # Find where p fits
        for i in range(len(anchor_points) - 1):
            p1, v1 = anchor_points[i]
            p2, v2 = anchor_points[i + 1]
            if p1 <= p <= p2:
                # Linear interpolation
                ratio = (p - p1) / (p2 - p1)
                return v1 + ratio * (v2 - v1)

        # Should never reach here if anchor_points cover [0,1]
        return 4.0

    speech_finished_cache = {}

    def is_speech_finished(text):
        # Returns a probability of completeness
        # Use cache if available
        if text in speech_finished_cache:
            return speech_finished_cache[text]

        prob_complete = get_completion_probability(
            text, classification_model, tokenizer, device, max_length
        )
        speech_finished_cache[text] = prob_complete
        return prob_complete

    if os.name == "nt" and (3, 8) <= sys.version_info < (3, 99):
        from torchaudio._extension.utils import _init_dll_path

        _init_dll_path()

    colorama.init()

    live = Live(console=console, refresh_per_second=10, screen=False)
    live.start()

    text_queue = queue.Queue()

    full_sentences = []
    rich_text_stored = ""
    recorder = None
    displayed_text = ""
    text_time_deque = deque()
    texts_without_punctuation = []
    relisten_on_abrupt_stop = True
    abrupt_stop = False
    prev_text = ""

    # Speaker diarization variables
    current_speaker = "Unknown"
    similarity_threshold = 0.5  # Adjust according to your needs
    audio_buffer = deque(maxlen=48000)  # For 3 seconds of audio at 16kHz
    audio_buffer_lock = threading.Lock()
    speech_audio_buffer = []
    is_speech = False  # Indicates if currently in speech segment

    process_embeddings_event = Event()

    # Load speaker embeddings (initialization)
    speaker_files = {
        "Adomas": "./Adomas.wav",
        # "Caedrel": "./Caedrel.wav",
        # "rekrap2": "./rekrap2.wav",
        "Zach": "./Zach.wav",
    }

    # Load ReDimNet model for embeddings
    model_name = "S"
    train_type = "ft_mix"
    dataset = "vb2+vox2+cnc"

    encoder = torch.hub.load(
        "IDRnD/ReDimNet",
        "ReDimNet",
        model_name=model_name,
        train_type=train_type,
        dataset=dataset,
    )
    encoder.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)

    def compute_embedding_from_file(filename):
        waveform, sr = torchaudio.load(f"./recordings/{filename}")
        # Resample if necessary to 16kHz
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
            sr = 16000
        waveform = waveform.to(device)
        with torch.no_grad():
            embedding = encoder(waveform)
        embedding_vector = embedding.squeeze(0).cpu().numpy()
        return embedding_vector

    speaker_embeddings = {}
    for speaker_name, file_path in speaker_files.items():
        embedding_vector = compute_embedding_from_file(file_path)
        speaker_embeddings[speaker_name] = embedding_vector

    def preprocess_text(text):
        text = text.lstrip()
        if text.startswith("..."):
            text = text[3:]
        text = text.lstrip()
        if text:
            text = text[0].upper() + text[1:]
        return text

    def text_detected(text):
        # Get the current speaker
        speaker = current_speaker
        text_queue.put((text, speaker))

    def ends_with_string(text: str, s: str):
        if text.endswith(s):
            return True
        if len(text) > 1 and text[:-1].endswith(s):
            return True
        return False

    def sentence_end(text: str):
        if text and text[-1] in sentence_end_marks:
            return True
        return False

    def additional_pause_based_on_words(text):
        word_count = len(text.split())
        pauses = {
            0: 0.35,
            1: 0.3,
            2: 0.25,
            3: 0.2,
            4: 0.15,
            5: 0.1,
            6: 0.05,
        }
        return pauses.get(word_count, 0.0)

    def strip_ending_punctuation(text):
        """Remove trailing periods and ellipses from text."""
        text = text.rstrip()
        for char in sentence_end_marks:
            text = text.rstrip(char)
        return text

    def get_suggested_whisper_pause(text):
        if ends_with_string(text, "..."):
            return ellipsis_pause
        elif ends_with_string(text, "."):
            return punctuation_pause
        elif ends_with_string(text, "!"):
            return exclamation_pause
        elif ends_with_string(text, "?"):
            return question_pause
        else:
            return unknown_sentence_detection_pause

    def find_stereo_mix_index():
        import pyaudio

        audio = pyaudio.PyAudio()
        devices_info = ""
        for i in range(audio.get_device_count()):
            dev = audio.get_device_info_by_index(i)
            devices_info += (
                f"{dev['index']}: {dev['name']} (hostApi: {dev['hostApi']})\n"
            )

            if (
                LOOPBACK_DEVICE_NAME.lower() in dev["name"].lower()
                and dev["hostApi"] == LOOPBACK_DEVICE_HOST_API
            ):
                return dev["index"], devices_info

        return None, devices_info

    def find_matching_texts(texts_without_punctuation):
        """
        Find entries where text_without_punctuation matches the last entry,
        going backwards until the first non-match is found.

        Args:
            texts_without_punctuation: List of tuples (original_text, stripped_text)

        Returns:
            List of tuples (original_text, stripped_text) matching the last entry's stripped text,
            stopping at the first non-match
        """
        if not texts_without_punctuation:
            return []

        # Get the stripped text from the last entry
        last_stripped_text = texts_without_punctuation[-1][1]

        matching_entries = []

        # Iterate through the list backwards
        for entry in reversed(texts_without_punctuation):
            original_text, stripped_text = entry

            # If we find a non-match, stop
            if stripped_text != last_stripped_text:
                break

            # Add the matching entry to our results
            matching_entries.append((original_text, stripped_text))

        # Reverse the results to maintain original order
        matching_entries.reverse()

        return matching_entries

    def process_queue():
        global recorder, full_sentences, prev_text, displayed_text, rich_text_stored, text_time_deque, abrupt_stop, rapid_sentence_end_detection, last_prob_complete, last_suggested_pause, last_pause
        while True:
            text = None  # Initialize text to ensure it's defined

            try:
                # Attempt to retrieve the first item, blocking with timeout
                (text, speaker) = text_queue.get(timeout=1)
            except queue.Empty:
                continue  # No item retrieved, continue the loop

            if text is None:
                # Exit signal received
                break

            # Drain the queue to get the latest text
            try:
                while True:
                    latest_item = text_queue.get_nowait()
                    if latest_item[0] is None:
                        text = None
                        break
                    text, speaker = latest_item
            except queue.Empty:
                pass  # No more items to retrieve

            if text is None:
                # Exit signal received after draining
                break

            text = preprocess_text(text)
            current_time = time.time()
            text_time_deque.append((current_time, text))

            # get text without ending punctuation
            text_without_punctuation = strip_ending_punctuation(text)

            texts_without_punctuation.append((text, text_without_punctuation))

            matches = find_matching_texts(texts_without_punctuation)

            added_pauses = 0
            contains_ellipses = False
            for i, match in enumerate(matches):
                same_text, stripped_punctuation = match
                suggested_pause = get_suggested_whisper_pause(same_text)
                added_pauses += suggested_pause
                if ends_with_string(same_text, "..."):
                    contains_ellipses = True

            avg_pause = added_pauses / len(matches) if len(matches) > 0 else 0
            suggested_pause = avg_pause

            prev_text = text
            import string

            transtext = text.translate(str.maketrans("", "", string.punctuation))

            # **Stripping Trailing Non-Alphabetical Characters**
            # Instead of removing all punctuation, we only strip trailing non-alphabetic chars.
            # Use regex to remove trailing non-alphabetic chars:
            cleaned_for_model = re.sub(r"[^a-zA-Z]+$", "", transtext)

            prob_complete = is_speech_finished(cleaned_for_model)

            # Interpolate rapid_sentence_end_detection based on prob_complete
            new_detection = interpolate_detection(prob_complete)

            pause = (new_detection + suggested_pause) * detection_speed

            # Optionally, you can log this information for debugging
            if IS_DEBUG:
                print(
                    f"Prob: {prob_complete:.2f}, "
                    f"whisper {suggested_pause:.2f}, "
                    f"model {new_detection:.2f}, "
                    f"final {pause:.2f} | {transtext} "
                )

            recorder.post_speech_silence_duration = pause

            # Remove old entries
            while (
                text_time_deque
                and text_time_deque[0][0]
                < current_time - hard_break_even_on_background_noise
            ):
                text_time_deque.popleft()

            # Check for abrupt stops (background noise)
            if len(text_time_deque) >= hard_break_even_on_background_noise_min_texts:
                texts = [t[1] for t in text_time_deque]
                first_text = texts[0]
                last_text = texts[-1]
                similarity = SequenceMatcher(None, first_text, last_text).ratio()

                if (
                    similarity > hard_break_even_on_background_noise_min_similarity
                    and len(first_text) > hard_break_even_on_background_noise_min_chars
                ):
                    abrupt_stop = True
                    recorder.stop()

            # Build the display text
            rich_text = Text()

            # Add final transcriptions
            for i, (sentence, spk) in enumerate(full_sentences):
                speaker_text = f"[{spk}] "
                style = "bold green" if spk != "Unknown" else "bold red"
                rich_text += Text(speaker_text + sentence, style=style) + "\n"

            # Add interim transcription
            if text:
                speaker_text = f"[{speaker}] "
                # Display interim text in italics
                rich_text += Text(speaker_text + text, style="italic yellow")

            # Update the live display
            panel = Panel(
                rich_text,
                title=f"Transcription",
                border_style="bold green",
            )
            live.update(panel)

            displayed_text = rich_text.plain
            last_prob_complete = new_detection
            last_suggested_pause = suggested_pause
            last_pause = pause
            rich_text_stored = rich_text

            text_queue.task_done()

    def process_text(text):
        global recorder, full_sentences, prev_text, abrupt_stop, last_detection_pause, current_speaker
        last_prob_complete, last_suggested_pause, last_pause
        last_detection_pause = recorder.post_speech_silence_duration
        if IS_DEBUG:
            print(
                f"Model pause: {last_prob_complete:.2f}, Whisper pause: {last_suggested_pause:.2f}, final pause: {last_pause:.2f}, last_detection_pause: {last_detection_pause:.2f}"
            )

        recorder.post_speech_silence_duration = unknown_sentence_detection_pause
        text = preprocess_text(text)
        text = text.rstrip()
        text_time_deque.clear()
        if text.endswith("..."):
            text = text[:-2]

        # Append the text and the current speaker
        full_sentences.append((text, current_speaker))
        prev_text = ""

        text_detected("")

        if abrupt_stop:
            abrupt_stop = False
            if relisten_on_abrupt_stop:
                recorder.listen()
                recorder.start()
                if hasattr(recorder, "last_words_buffer"):
                    recorder.frames.extend(list(recorder.last_words_buffer))

    def on_speech_start():
        global speech_audio_buffer, is_speech
        speech_audio_buffer = []
        is_speech = True
        if IS_DEBUG:
            print("Speech started")
        process_embeddings_event.clear()

    def on_speech_end():
        global speech_audio_buffer, current_speaker, is_speech
        is_speech = False
        process_embeddings_event.set()
        if len(speech_audio_buffer) == 0:
            return  # No audio collected
        # Convert speech_audio_buffer to numpy array
        speech_audio_data = (
            np.concatenate(speech_audio_buffer).astype(np.float32) / 32768.0
        )
        # Convert to tensor
        audio_tensor = torch.from_numpy(speech_audio_data).unsqueeze(0)
        audio_tensor = audio_tensor.to(device)
        # Compute embedding
        with torch.no_grad():
            embedding = encoder(audio_tensor)
        embedding_vector = embedding.squeeze(0).cpu().numpy()
        # Compare to stored embeddings
        max_similarity = -1
        best_speaker = None
        for speaker_name, speaker_embedding in speaker_embeddings.items():
            similarity = np.dot(embedding_vector, speaker_embedding) / (
                np.linalg.norm(embedding_vector) * np.linalg.norm(speaker_embedding)
            )
            if similarity > max_similarity:
                max_similarity = similarity
                best_speaker = speaker_name
        if max_similarity > similarity_threshold:
            current_speaker = best_speaker
        else:
            current_speaker = "Unknown"
        if IS_DEBUG:
            print(f"Speech ended. Speaker: {current_speaker}")

    def process_audio_frames(data):
        global audio_buffer, speech_audio_buffer
        # Convert data to numpy array
        audio_data = np.frombuffer(data, dtype=np.int16)
        # Append to audio_buffer
        with audio_buffer_lock:
            audio_buffer.extend(audio_data)
        # If in speech, append to speech_audio_buffer
        if is_speech:
            speech_audio_buffer.append(audio_data)

    def embedding_worker():
        global current_speaker
        N_samples = 48000  # Last 3 seconds at 16kHz
        min_samples = 16000  # At least 1 second of audio to process
        previous_speaker = current_speaker  # Keep track of the last speaker
        while True:
            process_embeddings_event.wait()
            time.sleep(0.2)
            with audio_buffer_lock:
                if len(audio_buffer) >= min_samples:
                    last_N_samples = list(audio_buffer)[-N_samples:]
                else:
                    continue  # Not enough samples yet
            # Convert to numpy array
            audio_segment = (
                np.array(last_N_samples, dtype=np.float32) / 32768.0
            )  # Normalize int16 to float [-1,1]
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio_segment).unsqueeze(
                0
            )  # Add batch dimension
            # Move to device
            audio_tensor = audio_tensor.to(device)
            # Compute embedding
            with torch.no_grad():
                embedding = encoder(audio_tensor)
            embedding_vector = embedding.squeeze(0).cpu().numpy()
            # Compare to stored embeddings
            max_similarity = -1
            best_speaker = None
            for speaker_name, speaker_embedding in speaker_embeddings.items():
                similarity = np.dot(embedding_vector, speaker_embedding) / (
                    np.linalg.norm(embedding_vector) * np.linalg.norm(speaker_embedding)
                )
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_speaker = speaker_name
            # If similarity exceeds threshold, update current speaker
            live.console.print(f"Speaker: {best_speaker}, Similarity: {max_similarity}")
            if max_similarity > similarity_threshold:
                current_speaker = best_speaker
            else:
                current_speaker = "Unknown"
            # Detect speaker change
            if current_speaker != previous_speaker:
                speaker_changed(previous_speaker, current_speaker)
                previous_speaker = current_speaker  # Update previous speaker

    def speaker_changed(former_speaker, new_speaker):
        backdate_stop_seconds = 2  # Adjust based on your needs
        backdate_resume_seconds = 0.5  # Adjust based on your needs
        if IS_DEBUG:
            print(f"Speaker changed from {former_speaker} to {new_speaker}.")
        # Stop the recorder with backdating
        recorder.stop(
            backdate_stop_seconds=backdate_stop_seconds,
            backdate_resume_seconds=backdate_resume_seconds,
        )
        # Restart the recorder if necessary
        recorder.listen()
        recorder.start()
        # Extend frames with last_words_buffer if needed
        if hasattr(recorder, "last_words_buffer"):
            recorder.frames.extend(list(recorder.last_words_buffer))

    recorder_config = {
        "spinner": False,
        # "model": "deepdml/faster-whisper-large-v3-turbo-ct2",
        "model": "Systran/faster-whisper-tiny",
        #'realtime_model_type': 'medium.en',
        "realtime_model_type": "tiny.en",
        "language": "en",
        "silero_sensitivity": 0.4,
        "webrtc_sensitivity": 3,
        "post_speech_silence_duration": unknown_sentence_detection_pause,
        "min_length_of_recording": 1.1,
        "min_gap_between_recordings": 0,
        "enable_realtime_transcription": True,
        "realtime_processing_pause": 0.05,
        "on_realtime_transcription_update": text_detected,
        "silero_deactivity_detection": True,
        "early_transcription_on_silence": 0,
        "beam_size": 5,
        "beam_size_realtime": 1,
        "batch_size": 4,
        "realtime_batch_size": 4,
        "no_log_file": True,
        "initial_prompt_realtime": (
            "End incomplete sentences with ellipses.\n"
            "Examples:\n"
            "Complete: The sky is blue.\n"
            "Incomplete: When the sky...\n"
            "Complete: She walked home.\n"
            "Incomplete: Because he...\n"
        ),
        # "on_speech_start": on_speech_start,
        "on_vad_detect_start": on_speech_start,
        "on_vad_detect_stop": on_speech_end,
        "on_recorded_chunk": process_audio_frames,  # Capture audio frames
    }

    if EXTENDED_LOGGING:
        recorder_config["level"] = logging.DEBUG

    if USE_STEREO_MIX:
        device_index, devices_info = find_stereo_mix_index()
        if device_index is None:
            live.stop()
            console.print(
                "[bold red]Stereo Mix device not found. Available audio devices are:\n[/bold red]"
            )
            console.print(devices_info, style="red")
            sys.exit(1)
        else:
            recorder_config["input_device_index"] = device_index
            console.print(
                f"Using audio device index {device_index} for Stereo Mix.",
                style="green",
            )

    recorder = AudioToTextRecorder(**recorder_config)

    initial_text = Panel(
        Text("Say something...", style="cyan bold"),
        title="[bold yellow]Waiting for Input[/bold yellow]",
        border_style="bold yellow",
    )
    live.update(initial_text)

    # Start the embedding worker thread
    embedding_thread = threading.Thread(target=embedding_worker, daemon=True)
    embedding_thread.start()

    worker_thread = threading.Thread(target=process_queue, daemon=True)
    worker_thread.start()

    try:
        while True:
            recorder.text(process_text)
    except KeyboardInterrupt:
        text_queue.put((None, None))
        worker_thread.join()
        live.stop()
        console.print("[bold red]Transcription stopped by user. Exiting...[/bold red]")
        exit(0)
