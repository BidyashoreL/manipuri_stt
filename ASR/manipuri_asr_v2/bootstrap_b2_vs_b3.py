from pathlib import Path
import os
import io
import wave
import gc
import json
import re
import unicodedata
import random
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel
from pyctcdecode import build_ctcdecoder


# ============================================================
# B2 vs B3-NARROW PAIRED BOOTSTRAP EVALUATION
# ============================================================
#
# Purpose:
#   Compare B2 and B3-NARROW on the EXACT SAME validation set.
#
# Models:
#   B2 = FINAL_B2_MODEL.pt
#   B3 = best.pt (Epoch 11)
#
# Evaluation:
#   - Same validation CSV
#   - Same W2V-BERT 2.0
#   - Same vocabulary
#   - Same NFC + whitespace normalization
#   - Same unigram decoder
#   - Same Beam-500
#   - 10,000 paired bootstrap resamples
#
# IMPORTANT:
#   The official 410-sample test set is NOT USED.
#
# ============================================================


# ============================================================
# PATHS
# ============================================================

ROOT = Path.cwd()
ASR = ROOT.parent

TRAIN_CSV = ASR / "combined_dataset" / "train.csv"
VAL_CSV = ASR / "combined_dataset" / "val.csv"

ENCODER_MODEL = Path(r"E:\hf_models\w2v-bert-2.0")

B2_CHECKPOINT = (
    ASR
    / "checkpoints"
    / "v2_ctc"
    / "b2_partial_encoder"
    / "regularized"
    / "FINAL_B2_MODEL.pt"
)

B3_CHECKPOINT = (
    ASR
    / "checkpoints"
    / "v3_ctc"
    / "b3_narrow"
    / "best.pt"
)

VOCAB_PATH = (
    ASR
    / "checkpoints"
    / "v2_ctc"
    / "b2_partial_encoder"
    / "vocab.json"
)

UNIGRAM_PATH = (
    ROOT
    / "kenlm_experiment"
    / "manipuri_unigrams.txt"
)

OUTPUT_DIR = ROOT / "bootstrap_b2_vs_b3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_CSV = OUTPUT_DIR / "paired_predictions.csv"
RESULTS_JSON = OUTPUT_DIR / "bootstrap_results.json"


# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAMPLE_RATE = 16000
FEATURE_DIM = 1024
BOTTLENECK = 256

BATCH_SIZE = 1
NUM_WORKERS = 0

BEAM_WIDTH = 500

N_BOOTSTRAP = 10_000
SEED = 42

SPECIAL_BLANK = "<blank>"
SPECIAL_UNK = "<unk>"

EXPECTED_VAL_SIZE = 864
EXPECTED_VOCAB_SIZE = 60


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    if text is None:
        return ""

    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# AUDIO LOADING
# ============================================================

def decode_pcm(raw, sample_width):

    if sample_width == 1:
        audio = np.frombuffer(
            raw,
            dtype=np.uint8
        ).astype(np.float32)

        return (audio - 128.0) / 128.0

    if sample_width == 2:
        return (
            np.frombuffer(
                raw,
                dtype=np.int16
            ).astype(np.float32)
            / 32768.0
        )

    if sample_width == 3:

        b = np.frombuffer(
            raw,
            dtype=np.uint8
        )

        if len(b) % 3:
            raise ValueError(
                "Invalid 24-bit PCM byte length"
            )

        b = b.reshape(-1, 3)

        audio = (
            b[:, 0].astype(np.int32)
            | (b[:, 1].astype(np.int32) << 8)
            | (b[:, 2].astype(np.int32) << 16)
        )

        mask = audio & 0x800000
        audio = audio - (mask << 1)

        return (
            audio.astype(np.float32)
            / 8388608.0
        )

    if sample_width == 4:
        return (
            np.frombuffer(
                raw,
                dtype=np.int32
            ).astype(np.float32)
            / 2147483648.0
        )

    raise ValueError(
        f"Unsupported PCM sample width: {sample_width}"
    )


@lru_cache(maxsize=4)
def _read_parquet_cached(parquet_file):

    return pd.read_parquet(
        str(parquet_file),
        columns=["audio"]
    )


def load_parquet_audio(parquet_file, row_index):

    parquet_file = str(parquet_file)

    if not os.path.exists(parquet_file):
        raise FileNotFoundError(
            parquet_file
        )

    df = _read_parquet_cached(parquet_file)

    obj = df.iloc[int(row_index)]["audio"]

    if not isinstance(obj, dict):
        raise ValueError(
            f"Unexpected audio object: {type(obj)}"
        )

    raw_bytes = obj.get("bytes")

    if raw_bytes is None:
        raise ValueError(
            "IndicVoices audio object has no bytes"
        )

    with wave.open(
        io.BytesIO(raw_bytes),
        "rb"
    ) as wf:

        channels = wf.getnchannels()
        width = wf.getsampwidth()
        sr = wf.getframerate()
        frames = wf.getnframes()

        raw = wf.readframes(frames)

    audio = decode_pcm(
        raw,
        width
    )

    if channels > 1:
        audio = audio.reshape(
            -1,
            channels
        ).mean(axis=1)

    return audio, sr


def normalize_audio(audio, sample_rate):

    audio = torch.as_tensor(
        audio,
        dtype=torch.float32
    )

    if audio.ndim == 2:

        if audio.shape[0] <= 8:
            audio = audio.mean(dim=0)
        else:
            audio = audio.mean(dim=1)

    elif audio.ndim > 2:
        audio = audio.squeeze()

    if sample_rate != SAMPLE_RATE:

        audio = torchaudio.functional.resample(
            audio,
            sample_rate,
            SAMPLE_RATE
        )

    return audio.contiguous()


def resolve_audio(row):

    source = str(
        row.get("source", "")
    ).strip().lower()

    if source == "indicvoices":

        audio, sr = load_parquet_audio(
            row["parquet_file"],
            int(row["parquet_row"])
        )

        return normalize_audio(
            audio,
            sr
        )

    path = row.get(
        "audio_path",
        ""
    )

    if pd.isna(path):
        raise RuntimeError(
            f"Invalid audio_path for source={source}"
        )

    path = str(path).strip()

    if not path:
        raise RuntimeError(
            f"Empty audio_path for source={source}"
        )

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    array, sr = sf.read(
        path,
        dtype="float32",
        always_2d=False
    )

    return normalize_audio(
        np.asarray(array),
        sr
    )


# ============================================================
# VOCABULARY
# ============================================================

def load_vocab():

    with open(
        VOCAB_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        vocab = json.load(f)

    if len(vocab) != EXPECTED_VOCAB_SIZE:
        raise RuntimeError(
            f"Unexpected vocabulary size: "
            f"{len(vocab)}; expected "
            f"{EXPECTED_VOCAB_SIZE}"
        )

    if vocab.get(SPECIAL_BLANK) != 0:
        raise RuntimeError(
            "Vocabulary blank token is not index 0."
        )

    return vocab


def encode_text(text, vocab):

    return torch.tensor(
        [
            vocab.get(
                ch,
                vocab[SPECIAL_UNK]
            )
            for ch in normalize_text(text)
        ],
        dtype=torch.long
    )


# ============================================================
# DATASET
# ============================================================

class AudioCTCDataset(Dataset):

    def __init__(
        self,
        df,
        vocab
    ):

        self.df = df.reset_index(
            drop=True
        )

        self.vocab = vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        return {
            "audio": resolve_audio(row),
            "target": encode_text(
                row["text"],
                self.vocab
            ),
            "text": normalize_text(
                row["text"]
            ),
            "row_index": idx,
        }


def collate_fn(batch):

    max_audio = max(
        x["audio"].shape[0]
        for x in batch
    )

    audios = torch.zeros(
        len(batch),
        max_audio,
        dtype=torch.float32
    )

    lengths = []
    targets = []
    target_lengths = []
    texts = []
    row_indices = []

    for i, item in enumerate(batch):

        n = item["audio"].shape[0]

        audios[
            i,
            :n
        ] = item["audio"]

        lengths.append(n)

        targets.append(
            item["target"]
        )

        target_lengths.append(
            item["target"].shape[0]
        )

        texts.append(
            item["text"]
        )

        row_indices.append(
            item["row_index"]
        )

    return {
        "audio": audios,
        "audio_lengths": torch.tensor(
            lengths
        ),
        "targets": torch.cat(
            targets
        ),
        "target_lengths": torch.tensor(
            target_lengths
        ),
        "texts": texts,
        "row_indices": row_indices,
    }


# ============================================================
# MODEL
# ============================================================

class ResidualBottleneckAdapter(nn.Module):

    def __init__(
        self,
        dim=1024,
        bottleneck=256
    ):

        super().__init__()

        self.norm_in = nn.LayerNorm(dim)

        self.down = nn.Linear(
            dim,
            bottleneck
        )

        self.act = nn.GELU()

        self.up = nn.Linear(
            bottleneck,
            dim
        )

        self.norm_out = nn.LayerNorm(
            dim
        )

        nn.init.normal_(
            self.up.weight,
            mean=0.0,
            std=1e-3
        )

        nn.init.zeros_(
            self.up.bias
        )

    def forward(self, x):

        z = self.up(
            self.act(
                self.down(
                    self.norm_in(x)
                )
            )
        )

        return self.norm_out(
            x + z
        )


class B2Model(nn.Module):

    def __init__(
        self,
        vocab_size
    ):

        super().__init__()

        print(
            "[MODEL] Loading W2V-BERT 2.0..."
        )

        self.processor = (
            AutoFeatureExtractor.from_pretrained(
                ENCODER_MODEL,
                local_files_only=True
            )
        )

        self.encoder = (
            Wav2Vec2BertModel.from_pretrained(
                ENCODER_MODEL,
                dtype=torch.float32,
                local_files_only=True
            )
        )

        self.encoder.config.layerdrop = 0.0

        for p in self.encoder.parameters():
            p.requires_grad = False

        self.adapter = (
            ResidualBottleneckAdapter(
                FEATURE_DIM,
                BOTTLENECK
            )
        )

        self.ctc_head = nn.Linear(
            FEATURE_DIM,
            vocab_size
        )

        self.encoder.to(DEVICE)
        self.adapter.to(DEVICE)
        self.ctc_head.to(DEVICE)

        try:
            self.encoder.gradient_checkpointing_disable()
        except Exception:
            pass

    def forward(
        self,
        input_features
    ):

        hidden = self.encoder(
            input_features=input_features
        ).last_hidden_state

        return self.ctc_head(
            self.adapter(hidden)
        )


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def extract_state_dict(checkpoint):

    if not isinstance(checkpoint, dict):
        return checkpoint

    for key in [
        "model_state_dict",
        "state_dict",
        "model",
    ]:

        if key in checkpoint:

            state = checkpoint[key]

            if isinstance(state, dict):
                return state

    return checkpoint


def load_checkpoint(
    model,
    checkpoint_path,
    model_name
):

    print()
    print(
        f"[CHECKPOINT] Loading {model_name}:"
    )

    print(
        checkpoint_path
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            checkpoint_path
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    state_dict = extract_state_dict(
        checkpoint
    )

    missing, unexpected = (
        model.load_state_dict(
            state_dict,
            strict=False
        )
    )

    print(
        f"[LOAD] Missing keys    : "
        f"{len(missing)}"
    )

    print(
        f"[LOAD] Unexpected keys : "
        f"{len(unexpected)}"
    )

    if missing:
        print(
            "[LOAD] Missing key examples:"
        )

        for key in missing[:10]:
            print(
                "   ",
                key
            )

        raise RuntimeError(
            f"{model_name} checkpoint "
            "has missing model parameters."
        )

    if unexpected:
        print(
            "[WARNING] Unexpected key examples:"
        )

        for key in unexpected[:10]:
            print(
                "   ",
                key
            )

    epoch = None

    if isinstance(checkpoint, dict):
        epoch = checkpoint.get(
            "epoch"
        )

    if epoch is not None:
        print(
            f"[CHECKPOINT] Epoch: {epoch}"
        )

    print(
        f"[CHECKPOINT] {model_name}: PASS"
    )


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def make_features(
    processor,
    audios
):

    audio_list = [
        audio.cpu()
        .numpy()
        .astype(
            np.float32,
            copy=False
        )
        for audio in audios
    ]

    output = processor(
        audio_list,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True
    )

    features = output.input_features

    del audio_list
    del output

    return features


# ============================================================
# CTC GREEDY DECODING
# ============================================================

def decode_ids(
    ids,
    vocab
):

    inverse_vocab = {
        int(v): k
        for k, v in vocab.items()
    }

    blank = vocab[
        SPECIAL_BLANK
    ]

    previous = None
    output = []

    for token in ids:

        token = int(token)

        if token == blank:

            previous = None
            continue

        if token == previous:
            continue

        output.append(
            inverse_vocab.get(
                token,
                SPECIAL_UNK
            )
        )

        previous = token

    return "".join(output)


# ============================================================
# EDIT DISTANCE
# ============================================================

def levenshtein(
    a,
    b
):

    if len(a) < len(b):
        a, b = b, a

    previous = list(
        range(len(b) + 1)
    )

    for i, ca in enumerate(a, 1):

        current = [i]

        for j, cb in enumerate(b, 1):

            insertion = (
                current[-1] + 1
            )

            deletion = (
                previous[j] + 1
            )

            substitution = (
                previous[j - 1]
                + (ca != cb)
            )

            current.append(
                min(
                    insertion,
                    deletion,
                    substitution
                )
            )

        previous = current

    return previous[-1]


def utterance_cer(
    reference,
    hypothesis
):

    reference = normalize_text(
        reference
    )

    hypothesis = normalize_text(
        hypothesis
    )

    ref_chars = list(
        reference
    )

    hyp_chars = list(
        hypothesis
    )

    distance = levenshtein(
        ref_chars,
        hyp_chars
    )

    denominator = max(
        len(ref_chars),
        1
    )

    return distance / denominator


def utterance_wer(
    reference,
    hypothesis
):

    reference = normalize_text(
        reference
    )

    hypothesis = normalize_text(
        hypothesis
    )

    ref_words = reference.split()
    hyp_words = hypothesis.split()

    distance = levenshtein(
        ref_words,
        hyp_words
    )

    denominator = max(
        len(ref_words),
        1
    )

    return distance / denominator


# ============================================================
# MICRO CER / WER
# Same metric definition used by training validation.
# ============================================================

def corpus_metrics(
    references,
    hypotheses
):

    char_errors = 0
    char_total = 0

    word_errors = 0
    word_total = 0

    for reference, hypothesis in zip(
        references,
        hypotheses
    ):

        reference = normalize_text(
            reference
        )

        hypothesis = normalize_text(
            hypothesis
        )

        ref_chars = list(
            reference
        )

        hyp_chars = list(
            hypothesis
        )

        char_errors += levenshtein(
            ref_chars,
            hyp_chars
        )

        char_total += max(
            len(ref_chars),
            1
        )

        ref_words = reference.split()
        hyp_words = hypothesis.split()

        word_errors += levenshtein(
            ref_words,
            hyp_words
        )

        word_total += max(
            len(ref_words),
            1
        )

    return {
        "cer": (
            char_errors
            / max(char_total, 1)
        ),

        "wer": (
            word_errors
            / max(word_total, 1)
        ),

        "char_errors": char_errors,
        "char_total": char_total,

        "word_errors": word_errors,
        "word_total": word_total,
    }


# ============================================================
# DECODER
# ============================================================

def build_decoder(
    vocab
):

    labels = [
        token
        for token, index in sorted(
            vocab.items(),
            key=lambda x: x[1]
        )
    ]

    labels = [
        ""
        if token == SPECIAL_BLANK
        else token
        for token in labels
    ]

    if not UNIGRAM_PATH.exists():

        raise FileNotFoundError(
            UNIGRAM_PATH
        )

    with open(
        UNIGRAM_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        unigrams = [
            line.strip()
            for line in f
            if line.strip()
        ]

    print(
        f"[DECODER] labels={len(labels)}, "
        f"unigrams={len(unigrams):,}, "
        f"beam={BEAM_WIDTH}"
    )

    return build_ctcdecoder(
        labels=labels,
        unigrams=unigrams
    )


# ============================================================
# MODEL EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_model(
    checkpoint_path,
    model_name,
    val_loader,
    vocab,
    decoder
):

    print()
    print(
        "=" * 72
    )

    print(
        f"EVALUATING {model_name}"
    )

    print(
        "=" * 72
    )

    model = B2Model(
        len(vocab)
    )

    load_checkpoint(
        model,
        checkpoint_path,
        model_name
    )

    model.eval()

    references = []
    greedy_predictions = []
    beam_predictions = []
    row_indices = []

    total_samples = len(
        val_loader.dataset
    )

    for batch_number, batch in enumerate(
        val_loader,
        start=1
    ):

        features = make_features(
            model.processor,
            batch["audio"]
        ).to(DEVICE)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=(
                DEVICE.type == "cuda"
                and torch.cuda.is_bf16_supported()
            )
        ):

            logits = model(
                features
            )

        # Greedy
        greedy_ids = (
            logits
            .argmax(dim=-1)
            .cpu()
            .tolist()
        )

        # Beam decoder requires log probabilities
        log_probs = (
            logits
            .float()
            .log_softmax(-1)
            .cpu()
            .numpy()
        )

        for i, reference in enumerate(
            batch["texts"]
        ):

            reference = normalize_text(
                reference
            )

            greedy_prediction = normalize_text(
                decode_ids(
                    greedy_ids[i],
                    vocab
                )
            )

            beam_prediction = normalize_text(
                decoder.decode(
                    log_probs[i],
                    beam_width=BEAM_WIDTH
                )
            )

            references.append(
                reference
            )

            greedy_predictions.append(
                greedy_prediction
            )

            beam_predictions.append(
                beam_prediction
            )

            row_indices.append(
                int(batch["row_indices"][i])
            )

        if (
            batch_number % 50 == 0
            or batch_number == total_samples
        ):

            current = len(
                references
            )

            print(
                f"[{model_name}] "
                f"{current}/{total_samples} "
                f"samples processed",
                flush=True
            )

        del features
        del logits
        del log_probs

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    greedy_metrics = corpus_metrics(
        references,
        greedy_predictions
    )

    beam_metrics = corpus_metrics(
        references,
        beam_predictions
    )

    print()
    print(
        f"[{model_name}] RESULTS"
    )

    print(
        f"Greedy CER   : "
        f"{greedy_metrics['cer'] * 100:.4f}%"
    )

    print(
        f"Greedy WER   : "
        f"{greedy_metrics['wer'] * 100:.4f}%"
    )

    print(
        f"Beam-{BEAM_WIDTH} CER : "
        f"{beam_metrics['cer'] * 100:.4f}%"
    )

    print(
        f"Beam-{BEAM_WIDTH} WER : "
        f"{beam_metrics['wer'] * 100:.4f}%"
    )

    # Release model before loading the next one.
    del model

    gc.collect()

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "row_indices": row_indices,
        "references": references,
        "greedy_predictions": greedy_predictions,
        "beam_predictions": beam_predictions,
        "greedy_metrics": greedy_metrics,
        "beam_metrics": beam_metrics,
    }


# ============================================================
# PAIRED BOOTSTRAP
# ============================================================

def paired_bootstrap(
    references,
    b2_predictions,
    b3_predictions,
    n_bootstrap=N_BOOTSTRAP,
    seed=SEED
):

    rng = np.random.default_rng(
        seed
    )

    n = len(references)

    if n == 0:
        raise RuntimeError(
            "No validation samples available."
        )

    print()
    print(
        "=" * 72
    )

    print(
        "PAIRED BOOTSTRAP"
    )

    print(
        "=" * 72
    )

    print(
        f"Validation utterances : {n:,}"
    )

    print(
        f"Bootstrap resamples    : "
        f"{n_bootstrap:,}"
    )

    # --------------------------------------------------------
    # Observed full-validation metrics
    # --------------------------------------------------------

    b2_observed = corpus_metrics(
        references,
        b2_predictions
    )

    b3_observed = corpus_metrics(
        references,
        b3_predictions
    )

    observed_cer_delta = (
        b3_observed["cer"]
        - b2_observed["cer"]
    )

    observed_wer_delta = (
        b3_observed["wer"]
        - b2_observed["wer"]
    )

    # --------------------------------------------------------
    # Per-utterance errors
    # --------------------------------------------------------

    b2_char_errors = np.array(
        [
            levenshtein(
                list(normalize_text(r)),
                list(normalize_text(h))
            )
            for r, h in zip(
                references,
                b2_predictions
            )
        ],
        dtype=np.float64
    )

    b3_char_errors = np.array(
        [
            levenshtein(
                list(normalize_text(r)),
                list(normalize_text(h))
            )
            for r, h in zip(
                references,
                b3_predictions
            )
        ],
        dtype=np.float64
    )

    char_denominators = np.array(
        [
            max(
                len(normalize_text(r)),
                1
            )
            for r in references
        ],
        dtype=np.float64
    )

    b2_word_errors = np.array(
        [
            levenshtein(
                normalize_text(r).split(),
                normalize_text(h).split()
            )
            for r, h in zip(
                references,
                b2_predictions
            )
        ],
        dtype=np.float64
    )

    b3_word_errors = np.array(
        [
            levenshtein(
                normalize_text(r).split(),
                normalize_text(h).split()
            )
            for r, h in zip(
                references,
                b3_predictions
            )
        ],
        dtype=np.float64
    )

    word_denominators = np.array(
        [
            max(
                len(normalize_text(r).split()),
                1
            )
            for r in references
        ],
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Bootstrap distributions
    #
    # We resample utterance indices WITH replacement.
    # Both systems receive exactly the same sampled indices.
    # --------------------------------------------------------

    bootstrap_cer_deltas = np.empty(
        n_bootstrap,
        dtype=np.float64
    )

    bootstrap_wer_deltas = np.empty(
        n_bootstrap,
        dtype=np.float64
    )

    for b in range(
        n_bootstrap
    ):

        indices = rng.integers(
            0,
            n,
            size=n
        )

        b2_cer = (
            b2_char_errors[indices].sum()
            /
            char_denominators[indices].sum()
        )

        b3_cer = (
            b3_char_errors[indices].sum()
            /
            char_denominators[indices].sum()
        )

        b2_wer = (
            b2_word_errors[indices].sum()
            /
            word_denominators[indices].sum()
        )

        b3_wer = (
            b3_word_errors[indices].sum()
            /
            word_denominators[indices].sum()
        )

        bootstrap_cer_deltas[b] = (
            b3_cer - b2_cer
        )

        bootstrap_wer_deltas[b] = (
            b3_wer - b2_wer
        )

        if (
            (b + 1) % 1000 == 0
            or b + 1 == n_bootstrap
        ):

            print(
                f"[BOOTSTRAP] "
                f"{b + 1:,}/{n_bootstrap:,}"
            )

    # --------------------------------------------------------
    # 95% percentile confidence intervals
    # --------------------------------------------------------

    cer_ci = np.percentile(
        bootstrap_cer_deltas,
        [2.5, 97.5]
    )

    wer_ci = np.percentile(
        bootstrap_wer_deltas,
        [2.5, 97.5]
    )

    # --------------------------------------------------------
    # Bootstrap tail probabilities
    #
    # Positive delta means B3 has higher error.
    # Negative delta means B3 has lower error.
    #
    # We report a two-sided bootstrap tail probability.
    # This is an empirical bootstrap p-value, not a parametric
    # test.
    # --------------------------------------------------------

    cer_left = np.mean(
        bootstrap_cer_deltas <= 0
    )

    cer_right = np.mean(
        bootstrap_cer_deltas >= 0
    )

    wer_left = np.mean(
        bootstrap_wer_deltas <= 0
    )

    wer_right = np.mean(
        bootstrap_wer_deltas >= 0
    )

    cer_p = min(
        1.0,
        2.0 * min(
            cer_left,
            cer_right
        )
    )

    wer_p = min(
        1.0,
        2.0 * min(
            wer_left,
            wer_right
        )
    )

    # --------------------------------------------------------
    # Convert everything to percentage points.
    # --------------------------------------------------------

    result = {

        "n_validation_utterances": int(n),

        "n_bootstrap": int(
            n_bootstrap
        ),

        "seed": int(seed),

        "b2": {
            "cer": float(
                b2_observed["cer"]
            ),
            "wer": float(
                b2_observed["wer"]
            ),
            "cer_percent": float(
                b2_observed["cer"] * 100
            ),
            "wer_percent": float(
                b2_observed["wer"] * 100
            ),
        },

        "b3": {
            "cer": float(
                b3_observed["cer"]
            ),
            "wer": float(
                b3_observed["wer"]
            ),
            "cer_percent": float(
                b3_observed["cer"] * 100
            ),
            "wer_percent": float(
                b3_observed["wer"] * 100
            ),
        },

        "delta_b3_minus_b2": {
            "cer": float(
                observed_cer_delta
            ),
            "wer": float(
                observed_wer_delta
            ),
            "cer_percentage_points": float(
                observed_cer_delta * 100
            ),
            "wer_percentage_points": float(
                observed_wer_delta * 100
            ),
        },

        "bootstrap": {

            "cer_95_ci_percentage_points": [
                float(cer_ci[0] * 100),
                float(cer_ci[1] * 100),
            ],

            "wer_95_ci_percentage_points": [
                float(wer_ci[0] * 100),
                float(wer_ci[1] * 100),
            ],

            "cer_p_value_empirical": float(
                cer_p
            ),

            "wer_p_value_empirical": float(
                wer_p
            ),
        },
    }

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    seed_everything()

    print()
    print(
        "=" * 72
    )

    print(
        "MANIPURI ASR — B2 vs B3-NARROW"
    )

    print(
        "PAIRED BOOTSTRAP EVALUATION"
    )

    print(
        "=" * 72
    )

    print(
        f"[DEVICE] {DEVICE}"
    )

    if DEVICE.type == "cuda":

        print(
            "[GPU]",
            torch.cuda.get_device_name(0)
        )

        print(
            "[VRAM]",
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    # --------------------------------------------------------
    # Verify paths
    # --------------------------------------------------------

    required_paths = [
        TRAIN_CSV,
        VAL_CSV,
        ENCODER_MODEL,
        B2_CHECKPOINT,
        B3_CHECKPOINT,
        VOCAB_PATH,
        UNIGRAM_PATH,
    ]

    print()
    print(
        "[PATH CHECK]"
    )

    for path in required_paths:

        if not path.exists():

            raise FileNotFoundError(
                f"\nRequired path missing:\n{path}"
            )

        print(
            "[PASS]",
            path
        )

    # --------------------------------------------------------
    # Load validation CSV
    # --------------------------------------------------------

    val_df = pd.read_csv(
        VAL_CSV,
        encoding="utf-8-sig"
    )

    print()
    print(
        f"[DATA] Validation rows: "
        f"{len(val_df):,}"
    )

    if len(val_df) != EXPECTED_VAL_SIZE:

        raise RuntimeError(
            "\nValidation dataset size mismatch.\n"
            f"Expected: {EXPECTED_VAL_SIZE}\n"
            f"Found   : {len(val_df)}\n"
            "\n"
            "This script intentionally stops rather than "
            "silently comparing different validation sets."
        )

    # --------------------------------------------------------
    # Vocabulary
    # --------------------------------------------------------

    vocab = load_vocab()

    print(
        f"[DATA] Vocabulary size: "
        f"{len(vocab)}"
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    val_dataset = AudioCTCDataset(
        val_df,
        vocab
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=(
            DEVICE.type == "cuda"
        ),
    )

    print(
        "[DATA] DataLoader ready."
    )

    # --------------------------------------------------------
    # Decoder
    # --------------------------------------------------------

    decoder = build_decoder(
        vocab
    )

    # --------------------------------------------------------
    # Evaluate B2
    # --------------------------------------------------------

    b2_result = evaluate_model(
        B2_CHECKPOINT,
        "B2",
        val_loader,
        vocab,
        decoder
    )

    # --------------------------------------------------------
    # Evaluate B3
    # --------------------------------------------------------

    b3_result = evaluate_model(
        B3_CHECKPOINT,
        "B3-NARROW",
        val_loader,
        vocab,
        decoder
    )

    # --------------------------------------------------------
    # Verify identical validation ordering
    # --------------------------------------------------------

    if (
        b2_result["row_indices"]
        !=
        b3_result["row_indices"]
    ):

        raise RuntimeError(
            "B2 and B3 validation row ordering differs."
        )

    references = b2_result[
        "references"
    ]

    if references != b3_result[
        "references"
    ]:

        raise RuntimeError(
            "B2 and B3 reference ordering differs."
        )

    # --------------------------------------------------------
    # Save paired predictions
    # --------------------------------------------------------

    rows = []

    for i in range(
        len(references)
    ):

        reference = references[i]

        b2_hyp = b2_result[
            "beam_predictions"
        ][i]

        b3_hyp = b3_result[
            "beam_predictions"
        ][i]

        rows.append({

            "id": int(
                b2_result[
                    "row_indices"
                ][i]
            ),

            "reference": reference,

            "b2_prediction": b2_hyp,

            "b3_prediction": b3_hyp,

            "b2_cer": utterance_cer(
                reference,
                b2_hyp
            ),

            "b3_cer": utterance_cer(
                reference,
                b3_hyp
            ),

            "b2_wer": utterance_wer(
                reference,
                b2_hyp
            ),

            "b3_wer": utterance_wer(
                reference,
                b3_hyp
            ),
        })

    paired_df = pd.DataFrame(
        rows
    )

    paired_df.to_csv(
        PREDICTIONS_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print(
        "[OUTPUT] Paired predictions:"
    )

    print(
        PREDICTIONS_CSV
    )

    # --------------------------------------------------------
    # Bootstrap
    # --------------------------------------------------------

    bootstrap_result = paired_bootstrap(
        references,
        b2_result[
            "beam_predictions"
        ],
        b3_result[
            "beam_predictions"
        ],
        n_bootstrap=N_BOOTSTRAP,
        seed=SEED
    )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    final_result = {

        "configuration": {

            "validation_csv": str(
                VAL_CSV
            ),

            "validation_size": int(
                len(val_df)
            ),

            "b2_checkpoint": str(
                B2_CHECKPOINT
            ),

            "b3_checkpoint": str(
                B3_CHECKPOINT
            ),

            "encoder_model": str(
                ENCODER_MODEL
            ),

            "vocab_path": str(
                VOCAB_PATH
            ),

            "unigram_path": str(
                UNIGRAM_PATH
            ),

            "beam_width": int(
                BEAM_WIDTH
            ),

            "normalization": (
                "NFC + whitespace collapse"
            ),

            "official_test_set_used": False,
        },

        "results": bootstrap_result,
    }

    with open(
        RESULTS_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            final_result,
            f,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    b2 = bootstrap_result["b2"]
    b3 = bootstrap_result["b3"]
    delta = bootstrap_result[
        "delta_b3_minus_b2"
    ]
    boot = bootstrap_result[
        "bootstrap"
    ]

    print()
    print(
        "=" * 72
    )

    print(
        "FINAL BOOTSTRAP RESULT"
    )

    print(
        "=" * 72
    )

    print(
        f"Validation samples : "
        f"{len(val_df):,}"
    )

    print()

    print(
        "B2:"
    )

    print(
        f"  Beam-{BEAM_WIDTH} CER : "
        f"{b2['cer_percent']:.4f}%"
    )

    print(
        f"  Beam-{BEAM_WIDTH} WER : "
        f"{b2['wer_percent']:.4f}%"
    )

    print()

    print(
        "B3-NARROW:"
    )

    print(
        f"  Beam-{BEAM_WIDTH} CER : "
        f"{b3['cer_percent']:.4f}%"
    )

    print(
        f"  Beam-{BEAM_WIDTH} WER : "
        f"{b3['wer_percent']:.4f}%"
    )

    print()

    print(
        "B3 - B2:"
    )

    print(
        f"  CER delta : "
        f"{delta['cer_percentage_points']:+.4f} pp"
    )

    print(
        f"  WER delta : "
        f"{delta['wer_percentage_points']:+.4f} pp"
    )

    print()

    print(
        "95% Bootstrap CI:"
    )

    print(
        "  CER:",
        "["
        f"{boot['cer_95_ci_percentage_points'][0]:+.4f}, "
        f"{boot['cer_95_ci_percentage_points'][1]:+.4f}"
        "] pp"
    )

    print(
        "  WER:",
        "["
        f"{boot['wer_95_ci_percentage_points'][0]:+.4f}, "
        f"{boot['wer_95_ci_percentage_points'][1]:+.4f}"
        "] pp"
    )

    print()

    print(
        "Empirical bootstrap tail p-value:"
    )

    print(
        f"  CER p-value : "
        f"{boot['cer_p_value_empirical']:.6f}"
    )

    print(
        f"  WER p-value : "
        f"{boot['wer_p_value_empirical']:.6f}"
    )

    print()

    print(
        "[OUTPUT] Results JSON:"
    )

    print(
        RESULTS_JSON
    )

    print()

    print(
        "[OUTPUT] Paired predictions:"
    )

    print(
        PREDICTIONS_CSV
    )

    print()

    print(
        "Official 410-sample test set: NOT TOUCHED"
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()
