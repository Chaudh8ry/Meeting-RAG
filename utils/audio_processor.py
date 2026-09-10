import os              # Provides functions for interacting with the operating system (like creating folders)
import yt_dlp          # Open-source library to download audio/video from YouTube and other sites
from pydub import AudioSegment  # Library for audio manipulation (conversion, slicing, etc.)

# Define the folder where downloaded files will be saved
DOWNLOAD_DIR = 'downloads'
# Create the folder if it doesn’t already exist
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Function to download audio from a YouTube URL
def download_youtube_audio(url: str) -> str:
    # Define the output file path pattern (title + extension)
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    
    # Configuration options for yt-dlp
    ydl_opts = {
        "format": "bestaudio/best",   # Download the best available audio quality
        "outtmpl": output_path,       # Save file using the defined path pattern
        "postprocessors": [           # After download, run FFmpeg to convert audio
            {
                "key": "FFmpegExtractAudio",   # Use FFmpeg to extract audio
                "preferredcodec": "wav",       # Convert audio to WAV format
                "preferredquality": "192",     # Target quality (bitrate)
            }
        ],
        # "quiet": True,              # Suppress yt-dlp logs in console (optional)
    }

    # Create a yt-dlp instance with the options
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        # Download the video/audio and get metadata
        info = ydl.extract_info(url, download=True)
        # Prepare the filename and replace extensions with .wav
        filename = ydl.prepare_filename(info).replace(".webm", ".wav").replace(".m4a", ".wav")
    
    # Return the final filename (path to the saved audio file)
    return filename

# Example usage: download audio from a YouTube link and print the saved filename
data = download_youtube_audio("https://youtu.be/P-D8udHlb70")

def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV format using pydub."""
    # Define output filename with "_converted.wav" suffix
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    # Load the input file into an AudioSegment object
    audio = AudioSegment.from_file(input_path)
    # Convert audio to mono (1 channel) and set frame rate to 16kHz
    audio = audio.set_channels(1).set_frame_rate(16000)
    # Export the processed audio as WAV
    audio.export(output_path, format="wav")
    return output_path

# Convert the downloaded file to WAV format
data_final = convert_to_wav(data)

def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:
    """Split a WAV file into smaller chunks of fixed duration."""
    # Load the WAV file
    audio = AudioSegment.from_wav(wav_path)
    # Convert minutes to milliseconds
    chunk_ms = chunk_minutes * 60 * 1000

    chunks = []  # List to store paths of chunked files

    # Loop through audio in steps of chunk_ms
    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        # Extract chunk from start to start+chunk_ms
        chunk = audio[start : start + chunk_ms]
        # Define chunk filename
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        # Export chunk as WAV
        chunk.export(chunk_path, format="wav")
        # Add chunk path to list
        chunks.append(chunk_path)
    
    return chunks

def process_input(source: str) -> list:
    """Process input (YouTube URL or local file) into WAV chunks."""
    # Check if input is a YouTube link
    if source.startswith("http://youtu.be") or source.startswith("https://youtu.be/"):
        print("Detected YouTube URL. Downloading audio.....")
        wav_path = download_youtube_audio(source)
    else:
        # Otherwise treat as local file
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    # Split audio into chunks
    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready - {len(chunks)} chunk(s) created.")
    return chunks