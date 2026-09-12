from dotenv import load_dotenv
load_dotenv()
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
import os

print("Key Loaded:",os.getenv("SARVAM_API_KEY"))
print("CWD:",os.getcwd())

source = "https://youtu.be/ETbK2ceAUU4"
language = "hinglish"

chunks = process_input(source)
transcript = transcribe_all(chunks,language)

print("\n =========== TRANSCRIPT ============= \n")
print(transcript)