from pathlib import Path
import json
import re
import unicodedata
import argparse

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================================
# PATHS
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
ASR_DIR = SCRIPT_DIR.parent

# Official untouched test set
TEST_CSV = ASR_DIR / "combined_dataset" / "test.csv"
TEST_CACHE = ASR_DIR / "test_audio_cache"

# B2 regularized checkpoints
B2_DIR = (
    ASR_DIR
    / "checkpoints"
    / "v2_ctc"
    / "b2_partial_encoder"
    / "regularized"
)

# Automatically use CUDA if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 1
NUM_WORKERS = 0

SPECIAL_BLANK = "<blank>"
SPECIAL_UNK = "<unk>"


# ============================================================================
# TEXT NORMALIZATION
# ============================================================================

def normalize_text(text):
    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================================
# CER
# ============================================================================

def cer(ref, hyp):
    ref = list(normalize_text(ref))
    hyp = list(normalize_text(hyp))

    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0

    prev = list(range(len(hyp) + 1))

    for i, r in enumerate(ref, 1):
        curr = [i]

        for j, h in enumerate(hyp, 1):
            cost = 0 if r == h else 1

            curr.append(
                min(
                    curr[-1] + 1,       # insertion
                    prev[j] + 1,        # deletion
                    prev[j - 1] + cost # substitution
                )
            )

        prev = curr

    return prev[-1] / len(ref)


# ============================================================================
# WER
# ============================================================================

def wer(ref, hyp):
    ref = normalize_text(ref).split()
    hyp = normalize_text(hyp).split()

    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0

    prev = list(range(len(hyp) + 1))

    for i, r in enumerate(ref, 1):
        curr = [i]

        for j, h in enumerate(hyp, 1):
            cost = 0 if r == h else 1

            curr.append(
                min(
                    curr[-1] + 1,       # insertion
                    prev[j] + 1,        # deletion
                    prev[j - 1] + cost # substitution
                )
            )

        prev = curr

    return prev[-1] / len(ref)


# ============================================================================
# LOAD VOCABULARY
# ============================================================================

def load_vocab():

    candidates = [
        B2_DIR / "vocab.json",
        ASR_DIR
        / "checkpoints"
        / "v2_ctc"
        / "b2_partial_encoder"
        / "vocab.json",
    ]

    for path in candidates:

        if path.exists():

            vocab = json.loads(
                path.read_text(encoding="utf-8")
            )

            print(f"[VOCAB] {path}")
            print(f"[VOCAB] Size: {len(vocab)}")

            return vocab

    raise FileNotFoundError(
        "vocab.json not found"
    )


# ============================================================================
# LOAD MODEL
# ============================================================================

def load_model(vocab_size, checkpoint_name):

    from train_b2_partial_encoder import B2Model

    print("[MODEL] Creating B2 model...")

    model = B2Model(
        vocab_size=vocab_size
    )

    model_path = B2_DIR / checkpoint_name

    if not model_path.exists():

        raise FileNotFoundError(
            f"B2 checkpoint not found:\n{model_path}"
        )

    print(
        f"[MODEL] Loading: {model_path}"
    )

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.to(DEVICE)
    model.eval()

    print(
        "[MODEL] Checkpoint loaded successfully"
    )

    return model


# ============================================================================
# TEST DATASET
# ============================================================================

class TestDataset(Dataset):

    def __init__(self, df, vocab):

        self.df = df.reset_index(drop=True)
        self.vocab = vocab

    def __len__(self):

        return len(self.df)

    def encode_text(self, text):

        text = normalize_text(text)

        return torch.tensor(
            [
                self.vocab.get(
                    ch,
                    self.vocab[SPECIAL_UNK]
                )
                for ch in text
            ],
            dtype=torch.long,
        )

    def __getitem__(self, idx):

        cache_path = (
            TEST_CACHE
            / f"{idx:06d}.pt"
        )

        if not cache_path.exists():

            raise FileNotFoundError(
                f"Test cache file missing:\n{cache_path}"
            )

        payload = torch.load(
            cache_path,
            map_location="cpu",
            weights_only=True,
        )

        audio = payload["audio"].float()

        text = normalize_text(
            self.df.iloc[idx]["text"]
        )

        target = self.encode_text(text)

        return {
            "audio": audio,
            "target": target,
            "text": text,
            "index": idx,
        }


# ============================================================================
# COLLATE
# ============================================================================

def collate_fn(batch):

    max_audio = max(
        x["audio"].numel()
        for x in batch
    )

    audios = torch.zeros(
        len(batch),
        max_audio,
        dtype=torch.float32,
    )

    lengths = []

    targets = []
    texts = []
    indices = []

    for i, item in enumerate(batch):

        n = item["audio"].numel()

        audios[i, :n] = item["audio"]

        lengths.append(n)

        targets.append(
            item["target"]
        )

        texts.append(
            item["text"]
        )

        indices.append(
            item["index"]
        )

    return {
        "audio": audios,
        "audio_lengths": torch.tensor(
            lengths
        ),
        "targets": targets,
        "texts": texts,
        "indices": indices,
    }


# ============================================================================
# GREEDY CTC DECODER
# ============================================================================

def greedy_decode(logits, vocab):

    ids = (
        torch.argmax(
            logits,
            dim=-1
        )
        .detach()
        .cpu()
        .tolist()
    )

    blank_id = vocab[
        SPECIAL_BLANK
    ]

    output = []

    previous = None

    for idx in ids:

        if idx == blank_id:

            previous = idx
            continue

        if idx == previous:

            continue

        output.append(idx)

        previous = idx

    inverse_vocab = {
        int(v): k
        for k, v in vocab.items()
    }

    chars = []

    for idx in output:

        token = inverse_vocab.get(
            idx,
            SPECIAL_UNK
        )

        if token not in (
            SPECIAL_BLANK,
            SPECIAL_UNK,
        ):

            chars.append(token)

    return "".join(chars)


# ============================================================================
# FINAL OFFICIAL TEST
# ============================================================================

@torch.no_grad()
def main():

    # ------------------------------------------------------------------------
    # ARGUMENTS
    # ------------------------------------------------------------------------

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        default="FINAL_B2_MODEL.pt",
        help=(
            "Checkpoint filename inside "
            "the regularized checkpoint directory"
        ),
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------------
    # HEADER
    # ------------------------------------------------------------------------

    print()
    print("=" * 72)
    print("MANIPURI ASR V2 — B2 OFFICIAL TEST EVALUATION")
    print("=" * 72)

    print(
        "[IMPORTANT] Official held-out test set"
    )

    print(
        "[IMPORTANT] Test set is NOT used "
        "for training or model selection"
    )

    print(
        f"[DEVICE] {DEVICE}"
    )

    if DEVICE.type == "cuda":

        print(
            f"[GPU] {torch.cuda.get_device_name(0)}"
        )

    print()

    print(
        f"[CHECKPOINT] "
        f"{B2_DIR / args.checkpoint}"
    )

    print(
        f"[TEST CSV] {TEST_CSV}"
    )

    print(
        f"[TEST CACHE] {TEST_CACHE}"
    )

    # ------------------------------------------------------------------------
    # CHECK PATHS
    # ------------------------------------------------------------------------

    if not TEST_CSV.exists():

        raise FileNotFoundError(
            f"Test CSV not found:\n{TEST_CSV}"
        )

    if not TEST_CACHE.exists():

        raise FileNotFoundError(
            f"Test cache not found:\n{TEST_CACHE}"
        )

    # ------------------------------------------------------------------------
    # LOAD VOCAB
    # ------------------------------------------------------------------------

    vocab = load_vocab()

    if len(vocab) != 60:

        raise RuntimeError(
            f"Expected vocab size 60, "
            f"got {len(vocab)}"
        )

    # ------------------------------------------------------------------------
    # LOAD TEST CSV
    # ------------------------------------------------------------------------

    df = pd.read_csv(
        TEST_CSV
    )

    print(
        f"[TEST] Samples: {len(df)}"
    )

    if len(df) != 410:

        raise RuntimeError(
            f"Expected exactly 410 official "
            f"test samples, got {len(df)}"
        )

    # ------------------------------------------------------------------------
    # VERIFY CACHE
    # ------------------------------------------------------------------------

    cache_count = len(
        list(TEST_CACHE.glob("*.pt"))
    )

    print(
        f"[TEST CACHE] Files: {cache_count}"
    )

    if cache_count != 410:

        raise RuntimeError(
            f"Expected exactly 410 cached "
            f"audio files, got {cache_count}"
        )

    # ------------------------------------------------------------------------
    # DATASET / LOADER
    # ------------------------------------------------------------------------

    dataset = TestDataset(
        df,
        vocab
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )

    # ------------------------------------------------------------------------
    # LOAD MODEL
    # ------------------------------------------------------------------------

    model = load_model(
        len(vocab),
        args.checkpoint
    )

    processor = model.processor

    # ------------------------------------------------------------------------
    # METRIC ACCUMULATORS
    # ------------------------------------------------------------------------

    total_cer = 0.0
    total_wer = 0.0
    total_loss = 0.0

    results = []

    # ------------------------------------------------------------------------
    # EVALUATION
    # ------------------------------------------------------------------------

    print()
    print("=" * 72)
    print("RUNNING OFFICIAL 410-SAMPLE TEST")
    print("=" * 72)
    print()

    for n, batch in enumerate(
        loader,
        1
    ):

        audio = batch["audio"].to(
            DEVICE,
            non_blocking=True,
        )

        targets = (
            batch["targets"][0]
            .to(DEVICE)
        )

        # ------------------------------------------------------------
        # Convert audio to float32 NumPy
        # ------------------------------------------------------------

        audio_np = (
            audio[0]
            .detach()
            .cpu()
            .numpy()
            .astype(
                "float32",
                copy=False,
            )
        )

        inputs = None
        input_values = None

        try:

            inputs = processor(
                audio_np,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            )

            if n == 1:

                print(
                    "[DEBUG] Processor keys:",
                    list(inputs.keys())
                )

            if "input_features" in inputs:

                input_values = (
                    inputs["input_features"]
                    .to(DEVICE)
                )

            elif "input_values" in inputs:

                input_values = (
                    inputs["input_values"]
                    .to(DEVICE)
                )

            else:

                raise RuntimeError(
                    "Unexpected processor output "
                    f"keys: {list(inputs.keys())}"
                )

        finally:

            del inputs
            del audio_np

        # ------------------------------------------------------------
        # MODEL FORWARD
        # ------------------------------------------------------------

        outputs = model(
            input_values
        )

        logits = outputs

        # ------------------------------------------------------------
        # CTC LOSS
        # ------------------------------------------------------------

        log_probs = (
            F.log_softmax(
                logits,
                dim=-1
            )
            .transpose(0, 1)
        )

        input_lengths = torch.full(
            (1,),
            logits.shape[1],
            dtype=torch.long,
            device=DEVICE,
        )

        target_lengths = torch.tensor(
            [targets.numel()],
            dtype=torch.long,
            device=DEVICE,
        )

        loss = F.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=vocab[
                SPECIAL_BLANK
            ],
            reduction="mean",
            zero_infinity=True,
        )

        # ------------------------------------------------------------
        # GREEDY DECODING
        # ------------------------------------------------------------

        hypothesis = greedy_decode(
            logits[0],
            vocab,
        )

        reference = (
            batch["texts"][0]
        )

        # ------------------------------------------------------------
        # METRICS
        # ------------------------------------------------------------

        sample_cer = cer(
            reference,
            hypothesis,
        )

        sample_wer = wer(
            reference,
            hypothesis,
        )

        total_loss += loss.item()
        total_cer += sample_cer
        total_wer += sample_wer

        # ------------------------------------------------------------
        # STORE RESULT
        # ------------------------------------------------------------

        results.append(
            {
                "index": batch[
                    "indices"
                ][0],

                "reference": reference,

                "hypothesis": hypothesis,

                "cer": sample_cer,

                "wer": sample_wer,

                "loss": loss.item(),
            }
        )

        # ------------------------------------------------------------
        # CLEAN CUDA MEMORY
        # ------------------------------------------------------------

        if DEVICE.type == "cuda":

            torch.cuda.empty_cache()

        # ------------------------------------------------------------
        # SHOW FIRST 10 EXAMPLES
        # ------------------------------------------------------------

        if n <= 10:

            print()
            print(
                f"[{n}/{len(df)}]"
            )

            print(
                "REF :",
                reference
            )

            print(
                "HYP :",
                hypothesis
            )

            print(
                f"CER : {sample_cer:.4f}"
            )

            print(
                f"WER : {sample_wer:.4f}"
            )

        # ------------------------------------------------------------
        # PROGRESS
        # ------------------------------------------------------------

        if n % 25 == 0 or n == len(df):

            print(
                f"[{n}/{len(df)}] "
                f"CER={total_cer / n:.4f} "
                f"WER={total_wer / n:.4f} "
                f"LOSS={total_loss / n:.4f}"
            )

    # ------------------------------------------------------------------------
    # FINAL METRICS
    # ------------------------------------------------------------------------

    avg_loss = (
        total_loss / len(loader)
    )

    avg_cer = (
        total_cer / len(loader)
    )

    avg_wer = (
        total_wer / len(loader)
    )

    checkpoint_name = Path(
        args.checkpoint
    ).stem

    # ------------------------------------------------------------------------
    # SAVE RESULTS
    # ------------------------------------------------------------------------

    result_path = (
        B2_DIR
        / f"{checkpoint_name}_test_results.csv"
    )

    pd.DataFrame(
        results
    ).to_csv(
        result_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary = {

        "checkpoint": str(
            B2_DIR
            / args.checkpoint
        ),

        "samples": len(loader),

        "test_loss": avg_loss,

        "CER": avg_cer,

        "WER": avg_wer,

        "vocab_size": len(vocab),

        "dataset": "official_410_sample_test",

        "test_set_used_for_training": False,

        "test_set_used_for_model_selection": False,
    }

    summary_path = (
        B2_DIR
        / f"{checkpoint_name}_test_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------------
    # FINAL REPORT
    # ------------------------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "B2 REGULARIZED OFFICIAL TEST COMPLETE"
    )
    print("=" * 72)

    print(
        f"Samples         : {len(loader)}"
    )

    print(
        f"Test loss       : {avg_loss:.6f}"
    )

    print(
        f"CER             : {avg_cer:.4%}"
    )

    print(
        f"WER             : {avg_wer:.4%}"
    )

    print()

    print(
        f"Results : {result_path}"
    )

    print(
        f"Summary : {summary_path}"
    )

    print("=" * 72)

    print()
    print(
        "✓ FINAL MODEL EVALUATED ON OFFICIAL "
        "410-SAMPLE TEST SET"
    )

    print(
        "✓ TEST SET WAS NOT USED FOR TRAINING"
    )

    print(
        "✓ TEST SET WAS NOT USED FOR MODEL SELECTION"
    )

    print(
        "✓ FINAL TEST METRICS ARE NOW LOCKED"
    )

    print("=" * 72)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()