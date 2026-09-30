# -*- coding: utf-8 -*-
"""
Hozirgi livekit/turn-detector modelini o'zbekcha EOU datasetda baholash.
LoRA fine-tune'dan OLDINGI bazaviy ko'rsatkich — keyin solishtirish uchun.
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

HG_MODEL = "livekit/turn-detector"
MODEL_REVISION = "v0.4.1-intl"
MAX_HISTORY_TOKENS = 128


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text.lower())
    text = "".join(ch for ch in text
                   if not (unicodedata.category(ch).startswith("P") and ch not in ["'", "-"]))
    return re.sub(r"\s+", " ", text).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    args = ap.parse_args()

    onnx_path = hf_hub_download(HG_MODEL, "model_q8.onnx", subfolder="onnx",
                                revision=MODEL_REVISION, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(HG_MODEL, revision=MODEL_REVISION,
                                              local_files_only=True, truncation_side="left")
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    def predict(messages):
        ctx = []
        last = None
        for msg in messages:
            content = normalize_text(msg["content"])
            if not content:
                continue
            if last and last["role"] == msg["role"]:
                last["content"] += f" {content}"
            else:
                last = {"role": msg["role"], "content": content}
                ctx.append(last)
        convo = tokenizer.apply_chat_template(ctx, add_generation_prompt=False,
                                              add_special_tokens=False, tokenize=False)
        convo = convo[:convo.rfind("<|im_end|>")]
        inputs = tokenizer(convo, add_special_tokens=False, return_tensors="np",
                           max_length=MAX_HISTORY_TOKENS, truncation=True)
        out = session.run(None, {"input_ids": inputs["input_ids"].astype("int64")})
        return float(out[0].flatten()[-1])

    for path in args.files:
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        probs = np.array([predict(r["messages"]) for r in rows])
        labels = np.array([r["label"] for r in rows])
        pos, neg = probs[labels == 1], probs[labels == 0]

        # AUC (rank asosida, sklearn'siz)
        order = probs.argsort().argsort()
        auc = (order[labels == 1].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg))

        print(f"\n=== {path.name} ({len(rows)} misol) ===")
        print(f"AUC: {auc:.4f}")
        print(f"mean prob | tugagan (label=1):      {pos.mean():.4f}")
        print(f"mean prob | tugallanmagan (label=0): {neg.mean():.4f}")
        for thr in [0.011, 0.05, 0.1, 0.5]:
            tpr = (pos >= thr).mean()
            tnr = (neg < thr).mean()
            acc = ((probs >= thr) == labels).mean()
            print(f"thr={thr:<6} TPR={tpr:.3f}  TNR={tnr:.3f}  acc={acc:.3f}")

        # naqsh bo'yicha aniqlik (thr=0.011 — intl 'ru/tr' darajasidagi past threshold)
        thr = 0.011
        by_pat = {}
        for r, p in zip(rows, probs):
            ok = (p >= thr) == (r["label"] == 1)
            by_pat.setdefault(r["pattern"], []).append(ok)
        print("naqsh bo'yicha (thr=0.011):")
        for pat, oks in sorted(by_pat.items(), key=lambda x: np.mean(x[1])):
            print(f"  {pat:20s} acc={np.mean(oks):.2f}  (n={len(oks)})")


if __name__ == "__main__":
    main()
