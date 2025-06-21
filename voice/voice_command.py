# voice/voice_command.py
import sounddevice as sd
import vosk
import json

model = vosk.Model("model/vosk-model-small-en-us-0.15")  # Download from Vosk's GitHub

def listen_command(audio_array, sample_rate=16000):
    rec = vosk.KaldiRecognizer(model, sample_rate)
    rec.AcceptWaveform(audio_array.tobytes())
    result = json.loads(rec.Result())
    return result.get("text", "").strip()
