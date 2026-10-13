import os
import subprocess
import tempfile
from pathlib import Path
import pandas as pd
from faster_whisper import WhisperModel

from normalizer import Step1Normalizer
from phonetic_engine import ManipuriPhoneticEngine

JSON_MAP_PATH = Path(
    "/Users/bidyashorelourembam/intern_nit/emacare/data/transliteration_map.json"
)
BENCHMARK_DIR = Path(
    "/Users/bidyashorelourembam/intern_nit/research/manipuri_stt/benchmark/"
)
GROUND_TRUTH_PATH = BENCHMARK_DIR / "ground_truth.csv"
OUTPUT_CSV_PATH = "results_medium_phonetic_longform_experiment.csv"


def prepare_audio_profile(input_file: Path) -> str:
    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    filter_chain = "loudnorm,highpass=f=80,lowpass=f=8000"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_file),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-af",
        filter_chain,
        temp_wav,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return temp_wav


def main():
    print("[1/3] Initializing Normalization & Phonetic Dictionary Maps...")
    normalizer = Step1Normalizer(max_allowed_repeats=2)
    engine = ManipuriPhoneticEngine(JSON_MAP_PATH)

    print("[2/3] Loading Whisper Model (Medium/CPU/Int8)...")
    model = WhisperModel("medium", device="cpu", compute_type="int8")

    if not GROUND_TRUTH_PATH.exists():
        print(f"Aborting: Ground Truth data missing from path: {GROUND_TRUTH_PATH}")
        return

    gt_df = pd.read_csv(GROUND_TRUTH_PATH)
    eval_records = []

    print("[3/3] Commencing Long-Form Audio Pipeline Assessment Execution...\n")

    # EXPANDED: Now catches conversational fragments like "I don't know", "yeah", "it's"
    ENGLISH_STOPWORDS = {
        "the",
        "this",
        "that",
        "with",
        "from",
        "have",
        "they",
        "are",
        "ok",
        "get",
        "well",
        "disciplined",
        "good",
        "you",
        "thank",
        "i",
        "don",
        "dont",
        "don't",
        "know",
        "yes",
        "yeah",
        "no",
        "it",
        "its",
        "it's",
        "is",
        "was",
        "for",
        "and",
        "but",
        "so",
        "what",
        "how",
        "why",
        "who",
        "where",
        "just",
        "like",
    }

    for idx, row in gt_df.iterrows():
        audio_path = BENCHMARK_DIR / row["file"]
        if not audio_path.exists():
            continue

        processed_wav = None
        whisper_raw = ""
        normalized_output = ""
        phonetic_final = ""

        try:
            processed_wav = prepare_audio_profile(audio_path)

            segments, _ = model.transcribe(
                processed_wav,
                language="en",
                beam_size=5,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=1000),
                initial_prompt="eikhoi, marang, kamaina, changge, nungairi, pot, haina, thongba.",
                repetition_penalty=1.35,
                no_repeat_ngram_size=4,
                temperature=[0.0, 0.2, 0.4],
                compression_ratio_threshold=2.2,
                no_speech_threshold=0.35,
            )
            whisper_raw = " ".join([seg.text.strip() for seg in segments])

            # FAIL-SAFE CRITICAL: Handle complete acoustic dropouts/silence stalls
            if not whisper_raw.strip():
                segments, _ = model.transcribe(
                    processed_wav,
                    language="en",
                    beam_size=5,
                    condition_on_previous_text=False,
                    vad_filter=False,
                    initial_prompt="eikhoi, marang, kamaina.",
                    temperature=0.0,
                )
                whisper_raw = " ".join([seg.text.strip() for seg in segments])

            # FAIL-SAFE A: Intercept foreign script hallucinations
            if any(ord(char) > 0x017F for char in whisper_raw.strip()):
                segments, _ = model.transcribe(
                    processed_wav,
                    language="en",
                    beam_size=5,
                    condition_on_previous_text=False,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=1000),
                    initial_prompt="",
                    repetition_penalty=1.1,
                    temperature=0.0,
                )
                whisper_raw = " ".join([seg.text.strip() for seg in segments])

            # FAIL-SAFE B: Strict English translation circuit-breaker
            words_tokens = set(whisper_raw.lower().split())
            if len(words_tokens.intersection(ENGLISH_STOPWORDS)) >= 1:
                segments, _ = model.transcribe(
                    processed_wav,
                    language="en",
                    beam_size=7,
                    condition_on_previous_text=False,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=1000),
                    initial_prompt="eikhoi, marang, kamaina, changge, nungairi, pot, haina, thongba.",
                    repetition_penalty=1.4,
                    temperature=0.0,
                )
                whisper_raw = " ".join([seg.text.strip() for seg in segments])

            # Process Output Pipeline
            normalized_output = normalizer.process(whisper_raw)
            phonetic_final = engine.clean_sentence(normalized_output)

        except Exception as err:
            whisper_raw = f"PIPELINE ERROR: {err}"
            normalized_output = whisper_raw
            phonetic_final = whisper_raw

        finally:
            if processed_wav and os.path.exists(processed_wav):
                try:
                    os.remove(processed_wav)
                except OSError:
                    pass

        print(f"--- [SAMPLE {idx:02d} | {row['file']}] ---")
        print(f"GROUND TRUTH : {row['text']}")
        print(f"RAW WHISPER  : {whisper_raw}")
        print(f"NORMALIZED   : {normalized_output}")
        print(f"PHON_RESOLVED: {phonetic_final}\n")

        eval_records.append(
            {
                "file": row["file"],
                "ground_truth": row["text"],
                "whisper_raw_acoustics": whisper_raw,
                "normalized_intermediary": normalized_output,
                "standardized_phonetics_final": phonetic_final,
            }
        )

    pd.DataFrame(eval_records).to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Experiment processing successfully saved to disk at: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
