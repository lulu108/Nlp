# -*- coding: utf-8 -*-
"""
BiLSTM-CRF for BMES tagging.

Input format (each line):
sentence \t char1 char2 ... \t tag1 tag2 ... \t word1 word2 ...
"""

from pathlib import Path
from typing import List, Tuple, Dict
import random
from collections import Counter

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

STATES = ["B", "M", "E", "S"]
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
EOS_TOKEN = "<EOS>"

# ====== training config ======
EPOCHS = 30
BATCH_SIZE = 16
DROPOUT = 0.3
PATIENCE = 10
RANDOM_SEED = 42
CHAR_EMB_DIM = 300
BIGRAM_EMB_DIM = 64
HIDDEN_DIM = 256
LR = 1e-3
EMB_LR = 3e-4
WEIGHT_DECAY = 1e-4
PRETRAINED_EMB_PATH = Path("./P1/BiLSTMCRF/data/cc.zh.300.vec/cc.zh.300.vec")
# 重新裁剪PRETRAINED_EMB_PATH = Path("./P1/BiLSTMCRF/data/cc.zh.300.char_subset.vec")
MIN_LR = 1e-5
T_MAX = EPOCHS
ETA_MIN = 1e-5
FREEZE_CHAR_EMB_EPOCHS = 2
SEED_LIST = [42, 52, 62, 72, 82]
APPLY_BMES_REPAIR = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def load_samples(path: Path) -> List[Tuple[List[str], List[str]]]:
    samples: List[Tuple[List[str], List[str]]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split("\t")
        if len(parts) != 4:
            continue
        chars = parts[1].split()
        tags = parts[2].split()
        if len(chars) != len(tags):
            continue
        samples.append((chars, tags))
    return samples


def build_vocab(samples: List[Tuple[List[str], List[str]]]) -> Dict[str, int]:
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for chars, _ in samples:
        for ch in chars:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab


def build_bigram_vocab(samples: List[Tuple[List[str], List[str]]]) -> Dict[str, int]:
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for chars, _ in samples:
        for i, ch in enumerate(chars):
            nxt = chars[i + 1] if i + 1 < len(chars) else EOS_TOKEN
            bg = f"{ch}{nxt}"
            if bg not in vocab:
                vocab[bg] = len(vocab)
    return vocab


def load_pretrained_char_embeddings(
    emb_path: Path,
    char2id: Dict[str, int],
    emb_dim: int,
) -> Tuple[torch.Tensor, int]:
    emb = torch.empty(len(char2id), emb_dim)
    nn.init.normal_(emb, mean=0.0, std=0.1)
    emb[char2id[PAD_TOKEN]].zero_()

    if not emb_path.exists():
        return emb, 0

    matched = 0
    with emb_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if not parts:
                continue
            # Skip common header line: "num_tokens dim".
            if i == 0 and len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                continue
            if len(parts) != emb_dim + 1:
                continue

            token = parts[0]
            if token not in char2id:
                continue
            try:
                vec = torch.tensor([float(x) for x in parts[1:]], dtype=torch.float)
            except ValueError:
                continue
            emb[char2id[token]] = vec
            matched += 1

    return emb, matched


def build_tag_vocab() -> Tuple[Dict[str, int], Dict[int, str]]:
    tag2id = {t: i for i, t in enumerate(STATES)}
    id2tag = {i: t for t, i in tag2id.items()}
    return tag2id, id2tag


class BMESDataset(Dataset):
    def __init__(self, samples, char2id, bigram2id, tag2id):
        self.samples = samples
        self.char2id = char2id
        self.bigram2id = bigram2id
        self.tag2id = tag2id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chars, tags = self.samples[idx]
        x = [self.char2id.get(ch, self.char2id[UNK_TOKEN]) for ch in chars]
        b = []
        for i, ch in enumerate(chars):
            nxt = chars[i + 1] if i + 1 < len(chars) else EOS_TOKEN
            bg = f"{ch}{nxt}"
            b.append(self.bigram2id.get(bg, self.bigram2id[UNK_TOKEN]))
        y = [self.tag2id[t] for t in tags]
        return x, b, y


def collate_fn(batch):
    lengths = [len(x) for x, _, _ in batch]
    max_len = max(lengths)
    batch_size = len(batch)

    x_pad = torch.zeros((batch_size, max_len), dtype=torch.long)
    b_pad = torch.zeros((batch_size, max_len), dtype=torch.long)
    y_pad = torch.zeros((batch_size, max_len), dtype=torch.long)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    for i, (x, b, y) in enumerate(batch):
        l = len(x)
        x_pad[i, :l] = torch.tensor(x, dtype=torch.long)
        b_pad[i, :l] = torch.tensor(b, dtype=torch.long)
        y_pad[i, :l] = torch.tensor(y, dtype=torch.long)
        mask[i, :l] = True

    return x_pad, b_pad, y_pad, mask, lengths


class CRF(nn.Module):
    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))
        if num_tags != len(STATES):
            raise ValueError("This CRF implementation expects BMES tag set with 4 tags.")
        self.register_buffer("transition_mask", self._build_transition_mask())
        self.register_buffer("start_mask", self._build_start_mask())
        self.register_buffer("end_mask", self._build_end_mask())

    @staticmethod
    def _build_transition_mask() -> torch.Tensor:
        # Valid BMES transitions:
        # B -> M/E, M -> M/E, E -> B/S, S -> B/S
        idx = {t: i for i, t in enumerate(STATES)}
        mask = torch.zeros((len(STATES), len(STATES)), dtype=torch.bool)
        allowed = {
            "B": ["M", "E"],
            "M": ["M", "E"],
            "E": ["B", "S"],
            "S": ["B", "S"],
        }
        for prev, currs in allowed.items():
            for curr in currs:
                mask[idx[prev], idx[curr]] = True
        return mask

    @staticmethod
    def _build_start_mask() -> torch.Tensor:
        idx = {t: i for i, t in enumerate(STATES)}
        mask = torch.zeros(len(STATES), dtype=torch.bool)
        mask[idx["B"]] = True
        mask[idx["S"]] = True
        return mask

    @staticmethod
    def _build_end_mask() -> torch.Tensor:
        idx = {t: i for i, t in enumerate(STATES)}
        mask = torch.zeros(len(STATES), dtype=torch.bool)
        mask[idx["E"]] = True
        mask[idx["S"]] = True
        return mask

    def _apply_constraints(self):
        neg = -1e4
        transitions = self.transitions.masked_fill(~self.transition_mask, neg)
        start_transitions = self.start_transitions.masked_fill(~self.start_mask, neg)
        end_transitions = self.end_transitions.masked_fill(~self.end_mask, neg)
        return transitions, start_transitions, end_transitions

    def log_likelihood(self, emissions, tags, mask):
        log_num = self._compute_score(emissions, tags, mask)
        log_den = self._compute_log_partition(emissions, mask)
        return log_num - log_den

    def _compute_score(self, emissions, tags, mask):
        batch_size, seq_len, _ = emissions.size()
        transitions, start_transitions, end_transitions = self._apply_constraints()

        first_tag = tags[:, 0]
        score = start_transitions[first_tag]
        score += emissions[:, 0].gather(1, first_tag.unsqueeze(1)).squeeze(1)

        for i in range(1, seq_len):
            mask_i = mask[:, i]
            prev_tag = tags[:, i - 1]
            curr_tag = tags[:, i]

            transition_score = transitions[prev_tag, curr_tag]
            emission_score = emissions[:, i].gather(1, curr_tag.unsqueeze(1)).squeeze(1)

            score += (transition_score + emission_score) * mask_i

        seq_ends = mask.long().sum(1) - 1
        last_tags = tags.gather(1, seq_ends.unsqueeze(1)).squeeze(1)
        score += end_transitions[last_tags]
        return score

    def _compute_log_partition(self, emissions, mask):
        batch_size, seq_len, num_tags = emissions.size()
        transitions, start_transitions, end_transitions = self._apply_constraints()

        alpha = start_transitions + emissions[:, 0]

        for i in range(1, seq_len):
            scores = alpha.unsqueeze(2) + transitions.unsqueeze(0)
            scores = torch.logsumexp(scores, dim=1) + emissions[:, i]
            alpha = torch.where(mask[:, i].unsqueeze(1), scores, alpha)

        alpha = alpha + end_transitions
        return torch.logsumexp(alpha, dim=1)

    def decode(self, emissions, mask):
        batch_size, seq_len, num_tags = emissions.size()
        transitions, start_transitions, end_transitions = self._apply_constraints()

        score = start_transitions + emissions[:, 0]
        backpointers = []

        for i in range(1, seq_len):
            next_score = score.unsqueeze(2) + transitions.unsqueeze(0)
            best_score, best_tag = next_score.max(dim=1)
            best_score = best_score + emissions[:, i]
            score = torch.where(mask[:, i].unsqueeze(1), best_score, score)
            backpointers.append(best_tag)

        score = score + end_transitions
        _, best_last_tag = score.max(dim=1)

        seq_ends = mask.long().sum(1) - 1
        best_paths = []
        for b in range(batch_size):
            last_tag = best_last_tag[b].item()
            length = seq_ends[b].item()
            path = [last_tag]
            for bp in reversed(backpointers[:length]):
                last_tag = bp[b][last_tag].item()
                path.append(last_tag)
            path.reverse()
            best_paths.append(path)
        return best_paths


class BiLSTMCRF(nn.Module):
    def __init__(
        self,
        vocab_size,
        bigram_vocab_size,
        num_tags,
        char_emb_dim=128,
        bigram_emb_dim=64,
        hidden_dim=256,
        pad_id=0,
        dropout=0.3,
    ):
        super().__init__()
        self.char_embedding = nn.Embedding(vocab_size, char_emb_dim, padding_idx=pad_id)
        self.bigram_embedding = nn.Embedding(bigram_vocab_size, bigram_emb_dim, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            char_emb_dim + bigram_emb_dim,
            hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, num_tags)
        self.crf = CRF(num_tags)

    def forward(self, chars, bigrams, mask, tags=None):
        char_emb = self.dropout(self.char_embedding(chars))
        bigram_emb = self.dropout(self.bigram_embedding(bigrams))
        emb = torch.cat([char_emb, bigram_emb], dim=-1)
        lstm_out, _ = self.lstm(emb)
        lstm_out = self.dropout(lstm_out)
        emissions = self.fc(lstm_out)

        if tags is not None:
            nll = -self.crf.log_likelihood(emissions, tags, mask)
            return nll
        return self.crf.decode(emissions, mask)


def tags_to_spans(tags: List[str]) -> List[Tuple[int, int]]:
    spans = []
    start = None
    for i, tag in enumerate(tags):
        if tag == "S":
            if start is not None:
                spans.append((start, i))
                start = None
            spans.append((i, i + 1))
        elif tag == "B":
            if start is not None:
                spans.append((start, i))
            start = i
        elif tag == "M":
            if start is None:
                start = i
        elif tag == "E":
            if start is None:
                spans.append((i, i + 1))
            else:
                spans.append((start, i + 1))
                start = None
        else:
            if start is not None:
                spans.append((start, i))
                start = None
            spans.append((i, i + 1))
    if start is not None:
        spans.append((start, len(tags)))
    return spans


def repair_bmes_tags(tags: List[str]) -> List[str]:
    if not tags:
        return tags

    repaired = tags[:]
    valid = {"B", "M", "E", "S"}

    for i, t in enumerate(repaired):
        if t not in valid:
            repaired[i] = "S"

    if repaired[0] in {"M", "E"}:
        repaired[0] = "S"

    for i in range(1, len(repaired)):
        prev = repaired[i - 1]
        curr = repaired[i]

        if prev == "B" and curr in {"B", "S"}:
            repaired[i - 1] = "S"
            prev = "S"
        elif prev == "M" and curr in {"B", "S"}:
            repaired[i - 1] = "E"
            prev = "E"

        if prev in {"E", "S"} and curr in {"M", "E"}:
            repaired[i] = "B"

    if len(repaired) == 1 and repaired[0] in {"B", "M", "E"}:
        repaired[0] = "S"
    elif repaired[-1] in {"B", "M"}:
        repaired[-1] = "E"

    return repaired


def repair_pred_ids(pred_ids: List[int], id2tag: Dict[int, str], tag2id: Dict[str, int]) -> List[int]:
    pred_tags = [id2tag[t] for t in pred_ids]
    repaired_tags = repair_bmes_tags(pred_tags)
    return [tag2id[t] for t in repaired_tags]


def predict_sequences(model, dataloader, id2tag, tag2id, device, apply_repair=True):
    model.eval()
    pred_sequences: List[List[int]] = []
    gold_sequences: List[List[int]] = []

    with torch.no_grad():
        for x, b, y, mask, _ in dataloader:
            x = x.to(device)
            b = b.to(device)
            y = y.to(device)
            mask = mask.to(device)
            pred_paths = model(x, b, mask)

            for i in range(len(pred_paths)):
                length = int(mask[i].sum().item())
                pred = pred_paths[i][:length]
                if apply_repair:
                    pred = repair_pred_ids(pred, id2tag, tag2id)
                gold = y[i][:length].tolist()
                pred_sequences.append(pred)
                gold_sequences.append(gold)

    return pred_sequences, gold_sequences


def evaluate_sequences(pred_sequences, gold_sequences, id2tag):
    total = 0
    correct = 0
    num_tags = len(id2tag)
    confusion = [[0 for _ in range(num_tags)] for _ in range(num_tags)]
    word_tp = 0
    word_fp = 0
    word_fn = 0

    for pred, gold in zip(pred_sequences, gold_sequences):
        for t1, t2 in zip(gold, pred):
            total += 1
            if t1 == t2:
                correct += 1
            confusion[t1][t2] += 1

        gold_tags = [id2tag[t] for t in gold]
        pred_tags = [id2tag[t] for t in pred]
        gold_spans = set(tags_to_spans(gold_tags))
        pred_spans = set(tags_to_spans(pred_tags))
        word_tp += len(gold_spans & pred_spans)
        word_fp += len(pred_spans - gold_spans)
        word_fn += len(gold_spans - pred_spans)

    acc = correct / total if total > 0 else 0.0
    word_precision = word_tp / (word_tp + word_fp) if (word_tp + word_fp) > 0 else 0.0
    word_recall = word_tp / (word_tp + word_fn) if (word_tp + word_fn) > 0 else 0.0
    word_f1 = (
        2 * word_precision * word_recall / (word_precision + word_recall)
        if (word_precision + word_recall) > 0
        else 0.0
    )
    return acc, confusion, (word_precision, word_recall, word_f1)


def evaluate(model, dataloader, id2tag, tag2id, device, apply_repair=True):
    pred_sequences, gold_sequences = predict_sequences(
        model,
        dataloader,
        id2tag,
        tag2id,
        device,
        apply_repair=apply_repair,
    )
    return evaluate_sequences(pred_sequences, gold_sequences, id2tag)


def majority_vote_ensemble(all_pred_sequences: List[List[List[int]]]) -> List[List[int]]:
    if not all_pred_sequences:
        return []

    num_models = len(all_pred_sequences)
    num_samples = len(all_pred_sequences[0])
    ensemble_preds: List[List[int]] = []

    for sample_idx in range(num_samples):
        lengths = [len(all_pred_sequences[m][sample_idx]) for m in range(num_models)]
        min_len = min(lengths)
        voted_seq: List[int] = []
        for pos in range(min_len):
            votes = [all_pred_sequences[m][sample_idx][pos] for m in range(num_models)]
            cnt = Counter(votes)
            best_tag = sorted(cnt.items(), key=lambda x: (-x[1], x[0]))[0][0]
            voted_seq.append(best_tag)
        ensemble_preds.append(voted_seq)

    return ensemble_preds


def print_confusion_matrix(confusion, id2tag):
    tags = [id2tag[i] for i in range(len(id2tag))]
    header = "true\\pred\t" + "\t".join(tags)
    print(header)
    for i, tag in enumerate(tags):
        row = "\t".join(str(confusion[i][j]) for j in range(len(tags)))
        print(f"{tag}\t{row}")


def save_confusion_matrix_tsv(confusion, id2tag, output_path: Path) -> None:
    tags = [id2tag[i] for i in range(len(id2tag))]
    lines = ["true\\pred\t" + "\t".join(tags)]
    for i, tag in enumerate(tags):
        row = "\t".join(str(confusion[i][j]) for j in range(len(tags)))
        lines.append(f"{tag}\t{row}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_preferred_font(size: int):
    # Prefer readable CJK fonts on Windows; fallback to default font.
    font_candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    if PIL_AVAILABLE:
        for fp in font_candidates:
            try:
                return ImageFont.truetype(fp, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def _draw_centered_text(draw, box, text, font, fill):
    x0, y0, x1, y1 = box
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    tw = right - left
    th = bottom - top
    tx = x0 + (x1 - x0 - tw) / 2
    ty = y0 + (y1 - y0 - th) / 2
    draw.text((tx, ty), text, fill=fill, font=font)


def save_confusion_matrix_png(confusion, id2tag, output_path: Path) -> None:
    if not PIL_AVAILABLE:
        print("Pillow is not installed. Skip confusion matrix PNG export.")
        return

    tags = [id2tag[i] for i in range(len(id2tag))]
    n = len(tags)

    cell = 150
    left_margin = 180
    top_margin = 180
    right_margin = 60
    bottom_margin = 80
    width = left_margin + n * cell + right_margin
    height = top_margin + n * cell + bottom_margin

    img = Image.new("RGB", (width, height), "#f7f9fc")
    draw = ImageDraw.Draw(img)
    title_font = _load_preferred_font(34)
    label_font = _load_preferred_font(24)
    value_font = _load_preferred_font(20)
    pct_font = _load_preferred_font(17)

    total = sum(sum(row) for row in confusion)

    max_val = max(max(row) for row in confusion) if confusion else 1
    max_val = max(max_val, 1)

    draw.text((left_margin, 25), "Confusion Matrix", fill="#1b2a41", font=title_font)
    draw.text(
        (left_margin, 70),
        f"Rows=True Labels, Cols=Pred Labels, Total={total}",
        fill="#3f4a5a",
        font=label_font,
    )

    # Axis labels.
    draw.text((left_margin + (n * cell) // 2 - 60, top_margin - 60), "Predicted", fill="#1b2a41", font=label_font)
    draw.text((30, top_margin + (n * cell) // 2 - 10), "True", fill="#1b2a41", font=label_font)

    for i, tag in enumerate(tags):
        cx = left_margin + i * cell
        cy = top_margin + i * cell
        _draw_centered_text(
            draw,
            (cx, top_margin - 45, cx + cell, top_margin - 10),
            tag,
            label_font,
            "#1b2a41",
        )
        _draw_centered_text(
            draw,
            (left_margin - 50, cy, left_margin - 10, cy + cell),
            tag,
            label_font,
            "#1b2a41",
        )

    for i in range(n):
        row_sum = sum(confusion[i])
        for j in range(n):
            val = confusion[i][j]
            ratio = val / max_val
            shade = int(245 - ratio * 190)
            color = (shade, shade + 5, 255)

            x0 = left_margin + j * cell
            y0 = top_margin + i * cell
            x1 = x0 + cell
            y1 = y0 + cell

            # Diagonal cells are emphasized with darker borders.
            border_color = "#2f3b52" if i == j else "#8ea0b8"
            border_width = 3 if i == j else 1
            draw.rectangle([x0, y0, x1, y1], fill=color, outline=border_color, width=border_width)

            pct = (val / row_sum * 100.0) if row_sum > 0 else 0.0
            _draw_centered_text(
                draw,
                (x0 + 4, y0 + 28, x1 - 4, y0 + 84),
                str(val),
                value_font,
                "#102030",
            )
            _draw_centered_text(
                draw,
                (x0 + 4, y0 + 88, x1 - 4, y1 - 10),
                f"{pct:.1f}%",
                pct_font,
                "#30435f",
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def train_one_seed(
    seed: int,
    train_ds,
    dev_ds,
    test_ds,
    char2id,
    bigram2id,
    tag2id,
    id2tag,
    device,
    output_dir: Path,
):
    set_seed(seed)

    data_generator = torch.Generator()
    data_generator.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        generator=data_generator,
    )
    dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = BiLSTMCRF(
        len(char2id),
        len(bigram2id),
        len(tag2id),
        char_emb_dim=CHAR_EMB_DIM,
        bigram_emb_dim=BIGRAM_EMB_DIM,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
    ).to(device)

    pretrained_emb, hit = load_pretrained_char_embeddings(
        PRETRAINED_EMB_PATH,
        char2id,
        CHAR_EMB_DIM,
    )
    model.char_embedding.weight.data.copy_(pretrained_emb.to(device))

    if PRETRAINED_EMB_PATH.exists():
        print(
            f"[seed={seed}] Loaded pretrained embeddings: {PRETRAINED_EMB_PATH} | "
            f"matched={hit}/{len(char2id)} ({hit / max(1, len(char2id)):.2%})"
        )
    else:
        print(f"[seed={seed}] Pretrained embeddings not found: {PRETRAINED_EMB_PATH}. Use random init.")

    # Freeze char embedding in early epochs, then unfreeze for task-specific fine-tuning.
    if FREEZE_CHAR_EMB_EPOCHS > 0:
        model.char_embedding.weight.requires_grad = False

    non_char_params = [
        p for n, p in model.named_parameters() if not n.startswith("char_embedding.")
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": non_char_params, "lr": LR, "weight_decay": WEIGHT_DECAY},
            {
                "params": [model.char_embedding.weight],
                "lr": EMB_LR,
                "weight_decay": WEIGHT_DECAY,
            },
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=T_MAX,
        eta_min=ETA_MIN,
    )

    best_word_f1 = -1.0
    best_acc = 0.0
    no_improve = 0
    best_model_path = output_dir / f"best_seed_{seed}.pt"

    for epoch in range(1, EPOCHS + 1):
        if FREEZE_CHAR_EMB_EPOCHS > 0 and epoch == FREEZE_CHAR_EMB_EPOCHS + 1:
            model.char_embedding.weight.requires_grad = True
            print(f"[seed={seed}] Unfreeze char embedding at epoch {epoch}.")

        model.train()
        total_loss = 0.0
        for x, b, y, mask, _ in train_loader:
            x = x.to(device)
            b = b.to(device)
            y = y.to(device)
            mask = mask.to(device)

            loss = model(x, b, mask, y).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()

        dev_acc, _, (wp, wr, wf1) = evaluate(
            model,
            dev_loader,
            id2tag,
            tag2id,
            device,
            apply_repair=APPLY_BMES_REPAIR,
        )
        scheduler.step()
        curr_main_lr = optimizer.param_groups[0]["lr"]
        curr_emb_lr = optimizer.param_groups[1]["lr"]

        improved = wf1 > best_word_f1 or (wf1 == best_word_f1 and dev_acc > best_acc)
        if improved:
            best_word_f1 = wf1
            best_acc = dev_acc
            no_improve = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            no_improve += 1

        print(
            f"[seed={seed}] Epoch {epoch} | loss={total_loss:.4f} | "
            f"dev_acc={dev_acc:.4f} | word_F1={wf1:.4f} | "
            f"best_word_F1={best_word_f1:.4f} | "
            f"lr_main={curr_main_lr:.6f} | lr_emb={curr_emb_lr:.6f} | no_improve={no_improve}"
        )

        if no_improve >= PATIENCE:
            print(f"[seed={seed}] Early stop at epoch {epoch} (patience={PATIENCE}).")
            break

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_acc, confusion, (wp, wr, wf1) = evaluate(
        model,
        test_loader,
        id2tag,
        tag2id,
        device,
        apply_repair=APPLY_BMES_REPAIR,
    )
    pred_sequences, gold_sequences = predict_sequences(
        model,
        test_loader,
        id2tag,
        tag2id,
        device,
        apply_repair=APPLY_BMES_REPAIR,
    )

    return {
        "seed": seed,
        "best_model_path": best_model_path,
        "test_acc": test_acc,
        "test_p": wp,
        "test_r": wr,
        "test_f1": wf1,
        "confusion": confusion,
        "test_preds": pred_sequences,
        "test_golds": gold_sequences,
    }


def main():
    data_dir = Path("./P1/BiLSTMCRF/data")
    train_path = data_dir / "train.txt"
    dev_path = data_dir / "dev.txt"
    test_path = data_dir / "test.txt"

    if not (train_path.exists() and dev_path.exists() and test_path.exists()):
        raise FileNotFoundError(
            "Data not found. Run: python P1/BiLSTMCRF/02_prepare_data.py"
        )

    train_samples = load_samples(train_path)
    dev_samples = load_samples(dev_path)
    test_samples = load_samples(test_path)

    char2id = build_vocab(train_samples)
    bigram2id = build_bigram_vocab(train_samples)
    tag2id, id2tag = build_tag_vocab()

    train_ds = BMESDataset(train_samples, char2id, bigram2id, tag2id)
    dev_ds = BMESDataset(dev_samples, char2id, bigram2id, tag2id)
    test_ds = BMESDataset(test_samples, char2id, bigram2id, tag2id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path("./P1/BiLSTMCRF/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_results = []
    for seed in SEED_LIST:
        print(f"\n===== Train seed={seed} =====")
        result = train_one_seed(
            seed,
            train_ds,
            dev_ds,
            test_ds,
            char2id,
            bigram2id,
            tag2id,
            id2tag,
            device,
            output_dir,
        )
        seed_results.append(result)
        print(
            f"[seed={seed}] Test tag accuracy: {result['test_acc']:.4f} | "
            f"P={result['test_p']:.4f} R={result['test_r']:.4f} F1={result['test_f1']:.4f}"
        )

    best_single = max(seed_results, key=lambda x: x["test_f1"])
    print("\n===== Best Single-Seed Evaluation =====")
    print(f"best_seed: {best_single['seed']}")
    print(f"Test tag accuracy: {best_single['test_acc']:.4f}")
    print(
        f"Test word-level: P={best_single['test_p']:.4f} "
        f"R={best_single['test_r']:.4f} F1={best_single['test_f1']:.4f}"
    )
    print_confusion_matrix(best_single["confusion"], id2tag)

    best_cm_png = output_dir / f"confusion_matrix_seed_{best_single['seed']}.png"
    save_confusion_matrix_png(best_single["confusion"], id2tag, best_cm_png)
    if PIL_AVAILABLE:
        print(f"Saved single-seed confusion matrix figure to: {best_cm_png}")
    else:
        best_cm_tsv = output_dir / f"confusion_matrix_seed_{best_single['seed']}.tsv"
        save_confusion_matrix_tsv(best_single["confusion"], id2tag, best_cm_tsv)
        print(f"Saved single-seed confusion matrix table to: {best_cm_tsv}")

    print("\n===== Ensemble Evaluation (Majority Vote) =====")
    all_seed_preds = [r["test_preds"] for r in seed_results]
    ensemble_preds = majority_vote_ensemble(all_seed_preds)
    gold_sequences = seed_results[0]["test_golds"]
    ens_acc, ens_confusion, (ens_p, ens_r, ens_f1) = evaluate_sequences(
        ensemble_preds,
        gold_sequences,
        id2tag,
    )
    print(f"Ensemble tag accuracy: {ens_acc:.4f}")
    print(f"Ensemble word-level: P={ens_p:.4f} R={ens_r:.4f} F1={ens_f1:.4f}")
    print_confusion_matrix(ens_confusion, id2tag)

    ens_cm_png = output_dir / "confusion_matrix_ensemble.png"
    save_confusion_matrix_png(ens_confusion, id2tag, ens_cm_png)
    if PIL_AVAILABLE:
        print(f"Saved ensemble confusion matrix figure to: {ens_cm_png}")
    else:
        ens_cm_tsv = output_dir / "confusion_matrix_ensemble.tsv"
        save_confusion_matrix_tsv(ens_confusion, id2tag, ens_cm_tsv)
        print(f"Saved ensemble confusion matrix table to: {ens_cm_tsv}")

if __name__ == "__main__":
    main()
