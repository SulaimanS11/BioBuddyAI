# voice/voice_command.py
import sounddevice as sd
import vosk
import json

model = vosk.Model("model/vosk-model-small-en-us-0.15")  # Download from Vosk's GitHub

def listen_command():
    print("Listening... (Say something like 'Eastern Massasauga')") # Mic is always on in the background to listen on your emotion and if sounding in distress, it will contact emergency services
    samplerate = 16000
    duration = 5  # seconds

    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()

    rec = vosk.KaldiRecognizer(model, samplerate)
    rec.AcceptWaveform(recording.tobytes())
    result = json.loads(rec.Result())
    return result.get("text", "").strip()
