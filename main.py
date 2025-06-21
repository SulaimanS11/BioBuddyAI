import sounddevice as sd
import numpy as np
from cam.live_detect import start_camera
from voice_filter import SpeakerIsolator
from voice_command import listen_command

def main():
    mode = input("Choose mode:\n1. Voice Command + Camera\n2. Just Image Path\n> ")

    if mode.strip() == '1':
        samplerate = 16000
        chunk_size = 4096  

        isolator = SpeakerIsolator(sample_rate=samplerate)

        def audio_callback(indata, frames, time, status): # processes each block of audio
            audio_in = indata[:, 0]
            filtered_audio, has_speech = isolator.isolate(audio_in)

            if has_speech:
                recognized_text = listen_command(
                    (filtered_audio * 32768).astype(np.int16), 
                    sample_rate=samplerate
                )
                if recognized_text:
                    print(f"Recognized Command: '{recognized_text}'")
                    raise sd.CallbackStop() # this stops the audio stream once a command is recognized

        print("Real-time voice recognition started. Speak now...")
        with sd.InputStream(callback=audio_callback, channels=1, samplerate=samplerate, blocksize=chunk_size): # listens to audio, stops listening when a command is recognized
            try:
                while True:
                    sd.sleep(100)
            except sd.CallbackStop:
                print("Audio processing stopped after command was recognized.")

    elif mode.strip() == '2':
        from classify import classify_image
        image_path = input("Enter path to image: ")
        result = classify_image(image_path)
        print(f"Result: {result}")
    else:
        print("Invalid option.")

if __name__ == "__main__":
    main()