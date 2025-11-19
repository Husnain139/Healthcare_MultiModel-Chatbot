
#Step1: Setup Audio recorder (using sounddevice - compatible with Python 3.14)
import sys
import logging
import sounddevice as sd
import soundfile as sf
import numpy as np
import os
import subprocess
import imageio_ffmpeg
from pathlib import Path
from dotenv import load_dotenv

# Get the directory where this script is located and load .env file
script_dir = Path(__file__).parent
env_path = script_dir / '.env'

# Load environment variables from .env file
load_dotenv(dotenv_path=env_path)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def record_audio(file_path, duration=10, sample_rate=44100):
    """
    Function to record audio from the microphone and save it as an MP3 file.

    Args:
    file_path (str): Path to save the recorded audio file.
    duration (int): Duration of recording in seconds (default: 10 seconds).
    sample_rate (int): Sample rate for recording (default: 44100 Hz).
    """
    try:
        logging.info(f"Recording for {duration} seconds...")
        logging.info("Start speaking now!")
        
        # Record audio
        audio_data = sd.rec(int(duration * sample_rate), 
                           samplerate=sample_rate, 
                           channels=1, 
                           dtype='float32')
        sd.wait()  # Wait until recording is finished
        
        logging.info("Recording complete.")
        
        # Save as temporary WAV file first
        temp_wav = file_path.replace('.mp3', '_temp.wav')
        sf.write(temp_wav, audio_data, sample_rate)
        
        # Convert WAV to MP3 using ffmpeg directly
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        command = [
            ffmpeg_path,
            '-i', temp_wav,
            '-codec:a', 'libmp3lame',
            '-b:a', '128k',
            '-y',  # Overwrite output file if it exists
            file_path
        ]
        
        subprocess.run(command, check=True, capture_output=True)
        
        # Clean up temporary WAV file
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        
        logging.info(f"Audio saved to {file_path}")
        return True

    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg conversion error: {e.stderr.decode()}")
        return False
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return False

# Step 1 is commented out - only Step 2 will run
audio_filepath = "patient_voice_test_for_patient.mp3"
# Uncomment the line below to record audio (Step 1):
record_audio(file_path=audio_filepath, duration=10)


#Step2: Setup Speech to text–STT–model for transcription
from groq import Groq

GROQ_API_KEY=os.environ.get("GROQ_API_KEY")
stt_model="whisper-large-v3"

def transcribe_with_groq(stt_model, audio_filepath, GROQ_API_KEY):
    client=Groq(api_key=GROQ_API_KEY)
    
    audio_file=open(audio_filepath, "rb")
    transcription=client.audio.transcriptions.create(
        model=stt_model,
        file=audio_file,
        language="en"
    )

    return transcription.text

# Run Step 2: Transcribe the audio file
if __name__ == "__main__":
    if not GROQ_API_KEY:
        logging.error("GROQ_API_KEY environment variable is not set!")
        print("Please set your GROQ_API_KEY environment variable.")
    elif not os.path.exists(audio_filepath):
        logging.error(f"Audio file not found: {audio_filepath}")
        print(f"Audio file '{audio_filepath}' does not exist. Please record audio first (Step 1).")
    else:
        logging.info("Starting transcription...")
        transcription_text = transcribe_with_groq(stt_model, audio_filepath, GROQ_API_KEY)
        logging.info("Transcription complete!")
        print("\n" + "="*50)
        print("TRANSCRIPTION:")
        print("="*50)
        print(transcription_text)
        print("="*50)
