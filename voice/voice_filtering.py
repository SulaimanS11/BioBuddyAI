# pylint: disable=trailing-whitespace
# voice/voice_isolation.py - Pure Speaker Isolation (No Voice Recognition)
"""
Simple Speaker Isolation for Pi 4b
"""
from voice_filter import SpeakerIsolator
import numpy as np
import scipy.signal

class SpeakerIsolator:
    """speaker isolation for clear speech (removes low speech)"""
    
    def __init__(self, sample_rate=16000):
        # Speech bandpass filter (100Hz - 6000Hz)
        nyquist = sample_rate * 0.5
        low, high = 100.0 / nyquist, 6000.0 / nyquist
        self.sos = scipy.signal.butter(4, [low, high], btype='band', output='sos')
        self.zi = scipy.signal.sosfilt_zi(self.sos)
        
        # Simple noise gate
        self.noise_floor = 0.01
        self.speech_threshold = 0.03
        
    def isolate(self, audio_chunk):
        """
        Isolate speaker from background noise
        
        Args:
            audio_chunk: Raw audio (int16 or float32)
            
        Returns:
            (clean_audio, is_speech): Tuple of cleaned audio and speech detection
        """
        if len(audio_chunk) == 0:
            return audio_chunk, False
            
        if audio_chunk.dtype == np.int16:
            audio = audio_chunk.astype(np.float32) / 32768.0
        else:
            audio = audio_chunk.astype(np.float32)
        
        # 1. Bandpass filter for speech frequencies
        filtered, self.zi = scipy.signal.sosfilt(self.sos, audio, zi=self.zi)
        
        # 2. sensitive voice activity detection
        energy = np.sqrt(np.mean(filtered ** 2))
        is_speech = energy > 0.02  # Higher value = less sensitive activation
        
        # 3. reduce non-speech audio
        if not is_speech:
            filtered *= 0.2  # Reduce background noise by 80%
            
        # Convert back to original format
        if audio_chunk.dtype == np.int16:
            output = np.clip(filtered * 32768.0, -32768, 32767).astype(np.int16)
        else:
            output = filtered
            
        return output, is_speech

def record_and_filter_audio(duration=5, sample_rate=16000):
    """
    Record raw audio and return filtered version
    
    Args:
        duration: Recording duration in seconds
        sample_rate: Audio sample rate
        
    Returns:
        (filtered_audio, has_speech): Filtered audio array and speech detection
    """
    import sounddevice as sd
    
    print("Recording audio for filtering...")
    
    # Initialize filter
    isolator = SpeakerIsolator(sample_rate)
    
    # Record raw audio
    recording = sd.rec(int(duration * sample_rate), 
                      samplerate=sample_rate, 
                      channels=1, 
                      dtype='int16')
    sd.wait()
    
    # Apply speaker isolation filter
    filtered_audio, has_speech = isolator.isolate(recording.flatten())
    
    print(f"Filtering complete. Speech detected: {has_speech}")
    
    return filtered_audio, has_speech

# Testing functions (same as your original)
def test_isolation_with_playback():
    """Record, clean, and play back audio to test isolation quality"""
    import sounddevice as sd
    import time
    
    isolator = SpeakerIsolator()
    sample_rate = 16000
    duration = 5
    
    print("Recording original audio (5 seconds)...")
    print("Speak with background noise")
    
    # Record original audio
    original_audio = sd.rec(int(duration * sample_rate), 
                           samplerate=sample_rate, 
                           channels=1, 
                           dtype='float32')
    sd.wait()
    
    print("Recording complete!")
    
    # Process audio through isolation
    print("Cleaning audio...")
    clean_audio = []
    speech_count = 0
    
    # Process in chunks
    chunk_size = 1024
    for i in range(0, len(original_audio), chunk_size):
        chunk = original_audio[i:i+chunk_size, 0]
        cleaned_chunk, has_speech = isolator.isolate(chunk)
        clean_audio.extend(cleaned_chunk)
        if has_speech:
            speech_count += 1
    
    clean_audio = np.array(clean_audio, dtype='float32')
    total_chunks = len(original_audio) // chunk_size
    
    print(f"Speech detected in {speech_count}/{total_chunks} chunks")
    
    # Play back comparison with volume boost for computer speakers
    print("\nPlaying ORIGINAL audio...")
    sd.play(original_audio[:, 0] * 2.0, sample_rate)
    sd.wait()
    
    time.sleep(1)
    
    print("Playing CLEANED audio...")
    sd.play(clean_audio * 2.0, sample_rate)
    sd.wait()
    
    print("\nPlayback complete!")
    return original_audio[:, 0], clean_audio

def save_audio_comparison():
    """Save original and cleaned audio to files for analysis"""
    import soundfile as sf
    
    print("🎵 Recording and saving audio comparison...")
    original, cleaned = test_isolation_with_playback()
    
    # Save audio files
    sf.write('original_audio.wav', original, 16000)
    sf.write('cleaned_audio.wav', cleaned, 16000)
    
    print("Saved files:")
    print("  • original_audio.wav - Raw recording")
    print("  • cleaned_audio.wav - After speaker isolation")

def test_isolation():
    """Test the simple isolation with real-time monitoring"""
    import sounddevice as sd
    
    isolator = SpeakerIsolator()
    print("Real-time isolation test (10 seconds)...")
    print("Speak to hear cleaned audio in real-time")
    
    def callback(indata, outdata, frames, time, status):
        audio_in = indata[:, 0]  # Get mono channel
        clean_audio, has_speech = isolator.isolate(audio_in)
        outdata[:, 0] = clean_audio  # Output cleaned audio
        if outdata.shape[1] > 1:
            outdata[:, 1] = clean_audio
            
        if has_speech:
            print("Speaking!", end="", flush=True)  # Visual feedback
    
    with sd.Stream(callback=callback, channels=1):
        sd.sleep(10000)  # 10 seconds
    
    print("\nReal-time test complete!")

if __name__ == "__main__":
    print("Choose test mode:")
    print("1. Real-time isolation")
    print("2. Record -> Clean -> Playback comparison")
    print("3. Save audio files for analysis")
    print("4. Test audio recording and filtering")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "1":
        test_isolation()
    elif choice == "2":
        test_isolation_with_playback()
    elif choice == "3":
        save_audio_comparison()
    elif choice == "4":
        filtered_audio, has_speech = record_and_filter_audio()
        print(f"Filtered {len(filtered_audio)} audio samples, speech detected: {has_speech}")
    else:
        print("Invalid choice, running playback test...")
        test_isolation_with_playback()