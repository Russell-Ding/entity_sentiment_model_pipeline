#!/usr/bin/env python3
"""Reproduce the NER-head measurements quoted in docs/model_structure.tex (Section 5).

Three checks on a released checkpoint:
  1. CRF transition scores: are the -1e4 BIO constraints present, and how far did the
     transition / start / end scores move from torchcrf's uniform(-0.1, 0.1) init?
  2. Emission-vs-transition scale on real articles, and how often Viterbi decoding
     differs from per-token argmax (if never, the CRF layer is inert in practice).
  3. How rare labels at exactly +-1 are in a label split (context for the 0.95*tanh bound).

Usage:
    python3 scripts/analysis/inspect_ner_head.py \
        --checkpoint trained_model/v2.1_20260620/model.pt \
        --articles data/labeled/deepseek_t1/splits/val.jsonl --n-articles 3

Runs on CPU in well under a minute (articles are truncated to --max-length tokens).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

try:  # older transformers versions refuse torch.load without this
    import transformers.utils.import_utils as _tiu
    _tiu.check_torch_load_is_safe = lambda: None
    import transformers.modeling_utils as _tmu
    _tmu.check_torch_load_is_safe = lambda: None
except Exception:  # pragma: no cover
    pass

from models.pipeline import FinancialEntitySentimentModel  # noqa: E402
from training.preprocessing import LABEL_TO_ID, ID_TO_LABEL, SENTIMENT_ENTITY_TYPES  # noqa: E402
from transformers import LongformerTokenizerFast  # noqa: E402


def transition_report(state_dict: dict) -> None:
    labels = [ID_TO_LABEL[i] for i in range(len(ID_TO_LABEL))]
    T = state_dict["ner_head.crf.transitions"]
    S = state_dict["ner_head.crf.start_transitions"]
    E = state_dict["ner_head.crf.end_transitions"]
    invalid, valid = [], []
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            allowed = not b.startswith("I-") or a in (f"B-{b[2:]}", f"I-{b[2:]}")
            (valid if allowed else invalid).append(T[i, j].item())
    print("CRF transition scores")
    print(f"  invalid BIO transitions (into I-X from other tags): n={len(invalid)} "
          f"min={min(invalid):+.3f} max={max(invalid):+.3f}")
    print(f"  valid transitions:                                  n={len(valid)} "
          f"min={min(valid):+.3f} max={max(valid):+.3f}")
    print(f"  start scores: min={S.min():+.3f} max={S.max():+.3f}; "
          f"end scores: min={E.min():+.3f} max={E.max():+.3f}")
    print(f"  -1e4 constraints present: {bool((T < -1000).any())}  "
          f"(torchcrf init range is uniform(-0.1, 0.1))")


def emission_report(model, tokenizer, articles: list, max_length: int) -> None:
    print("Emission vs. transition scale (Viterbi vs. argmax agreement)")
    for art in articles:
        enc = tokenizer(art["text"], return_tensors="pt", max_length=max_length,
                        truncation=True, padding="max_length")
        with torch.no_grad():
            hidden = model.encoder(enc["input_ids"], enc["attention_mask"])
            out = model.ner_head(hidden, attention_mask=enc["attention_mask"])
        n = int(enc["attention_mask"].sum())
        logits = out["logits"][0, :n]
        top2 = logits.topk(2, dim=-1).values
        margin = top2[:, 0] - top2[:, 1]
        viterbi = out["predictions"][0, :n]
        agree = (logits.argmax(-1) == viterbi).float().mean().item()
        print(f"  {str(art.get('id', '?'))[:12]:12s} tokens={n:4d} emission std={logits.std():.2f} "
              f"|max|={logits.abs().max():.1f} median top1-top2 margin={margin.median():.2f} "
              f"viterbi==argmax on {agree * 100:.1f}% of tokens "
              f"({int((viterbi != 0).sum())} non-O tags)")


def label_extremes(path: Path, n_lines: int) -> None:
    total = extreme = 0
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= n_lines or not line.strip():
                continue
            for e in json.loads(line).get("entities", []):
                y = e.get("sentiment_score")
                if y is None or e.get("type") not in SENTIMENT_ENTITY_TYPES:
                    continue
                total += 1
                extreme += abs(y) >= 1.0
    print(f"Labels at exactly +-1 in {path.name} (first {n_lines} lines): "
          f"{extreme} of {total} scored entities ({100 * extreme / max(total, 1):.3f}%)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=str(PROJECT_ROOT / "trained_model/v2.1_20260620/model.pt"))
    ap.add_argument("--articles", default=str(PROJECT_ROOT / "data/labeled/deepseek_t1/splits/val.jsonl"))
    ap.add_argument("--n-articles", type=int, default=3)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--label-lines", type=int, default=4000, help="lines of --articles to scan for +-1 labels")
    args = ap.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model_state_dict"]
    transition_report(state)

    model = FinancialEntitySentimentModel(
        encoder_name="allenai/longformer-large-4096", hidden_size=1024, num_ner_labels=15,
        use_ner_head=True, use_coref_head=False, use_crf_ner=True,
        ner_label_to_id=LABEL_TO_ID, device="cpu", max_length=args.max_length,
    )
    print(f"constraint value written at construction: {model.ner_head.crf.transitions.min().item():.0f}")
    model.load_state_dict(state)
    model.eval()
    print(f"after load_state_dict: min transition = {model.ner_head.crf.transitions.min().item():+.3f} "
          f"(the checkpoint overwrites the constraints)")

    articles_path = Path(args.articles)
    if articles_path.exists():
        tokenizer = LongformerTokenizerFast.from_pretrained("allenai/longformer-large-4096")
        articles = []
        with open(articles_path) as f:
            for line in f:
                if line.strip():
                    articles.append(json.loads(line))
                if len(articles) >= args.n_articles:
                    break
        emission_report(model, tokenizer, articles, args.max_length)
        label_extremes(articles_path, args.label_lines)
    else:
        print(f"(no article file at {articles_path}; skipping the emission and label checks)")


if __name__ == "__main__":
    main()
