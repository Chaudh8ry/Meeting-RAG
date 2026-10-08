import whisper
import os
import json
from sarvamai import SarvamAI

# Configuration
WHISPER_MODEL = os.getenv("WHISPER_MODEL","small")

SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL","saaras:v3")
# SARVAM_LANGUAGE = os.getenv("SARVAM_LANGUAGE","hi-IN")

SARVAM_OUTPUT_DIR = "sarvam_outputs"
SARVAM_MAX_FILES_PER_JOB = 20 # max no. of files we can put in one batch

_whisper_model = None
_sarvam_client = None

# ---------------- Whisper (English) -------------------

def load_model():
    global _whisper_model
    if _whisper_model is None:
        print(f"Loading Whisper model: {WHISPER_MODEL}...")
        _whisper_model = whisper.load_model(WHISPER_MODEL)
        print("whisper model loaded successfully")
    return _whisper_model

# transcribing one chunk with whisper
def transcribe_chunk_whisper(chunk_path: str) -> str:
    model = load_model()
    result = model.transcribe(chunk_path,task = "transcribe")
    return result['text'] # returns only the actual transcript 'text' from dictionary

# ------------------- Sarvam (Hindi/Hinglish, via Batch API) ------------------
def get_sarvam_client() -> SarvamAI:
    global _sarvam_client
    if _sarvam_client is None:
        # Read this at client creation time so callers that load `.env` during
        # application startup are supported too.
        sarvam_api_key = os.getenv("SARVAM_API_KEY")
        if not sarvam_api_key:
            raise RuntimeError("SARVAM_API_KEY is not set in Environment/ .env")
        _sarvam_client = SarvamAI(api_subscription_key=sarvam_api_key)
    return _sarvam_client

def run_sarvam_batch_job(file_paths: list) -> dict:
    """
    Submit one batch job for up to 20 files, wait for it to finish,
    and return {file_path: transcript_text}.
    """
    client = get_sarvam_client()

    job = client.speech_to_text_job.create_job(
        model=SARVAM_MODEL,
        mode="translate",
        with_diarization=False, # for speaker separation
    )

    job.upload_files(file_paths=file_paths)
    job.start()

    print(f"  → Sarvam batch job submitted for {len(file_paths)} file(s), waiting...")
    job.wait_until_complete(poll_interval=5, timeout=900)

    file_results = job.get_file_results()
    for f in file_results.get("failed", []):
        print(f"  ✗ Sarvam failed on {f['file_name']}: {f.get('error_message')}")

    successful = file_results.get("successful", [])
    if not successful:
        return {}

    os.makedirs(SARVAM_OUTPUT_DIR, exist_ok=True)
    job.download_outputs(output_dir=SARVAM_OUTPUT_DIR)

    name_to_path = {os.path.basename(p): p for p in file_paths}
    results = {}
    for f in successful:
        input_name = f["file_name"]                 # e.g. "chunk_0.wav"
        original_path = name_to_path.get(input_name)
        if not original_path:
            continue
        # download_outputs() names files "{input_filename}.json"
        output_path = os.path.join(SARVAM_OUTPUT_DIR, f"{input_name}.json")
        with open(output_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        results[original_path] = data.get("transcript", "")

    return results


def transcribe_chunks_sarvam(chunk_paths: list) -> dict:
    """
    Transcribe a list of chunk files via Sarvam Batch API, respecting the
    20-files-per-job cap by splitting into multiple jobs if needed.
    Returns {chunk_path: transcript_text}, preserving nothing about order
    (caller should reassemble using the original chunk_paths list).
    """
    all_results = {}
    for i in range(0, len(chunk_paths), SARVAM_MAX_FILES_PER_JOB):
        batch = chunk_paths[i : i + SARVAM_MAX_FILES_PER_JOB]
        all_results.update(run_sarvam_batch_job(batch))
    return all_results


# ---------- Unified interface ----------

def transcribe_all(chunks: list, language: str = "english") -> str:
    engine = "Sarvam AI (batch)" if language.lower() == "hinglish" else "Whisper"
    print(f"Using {engine} for transcription.")

    if language.lower() == "hinglish":
        results = transcribe_chunks_sarvam(chunks)
        # Reassemble in original chunk order; skip any that failed.
        full_transcript = " ".join(results.get(c, "") for c in chunks)
    else:
        full_transcript = ""
        for i, chunk in enumerate(chunks):
            print(f"Transcribing chunk {i + 1}/{len(chunks)}...")
            full_transcript += transcribe_chunk_whisper(chunk) + " "

    print("Transcription complete.")
    return full_transcript.strip()
