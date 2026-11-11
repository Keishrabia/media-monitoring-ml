
import argparse
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from utils import DATA_PROCESSED, METRICS, MODELS, ensure_dirs

SEED = 42


class DS(torch.utils.data.Dataset):
    def __init__(self, enc, labels):
        self.enc, self.labels = enc, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[i])
        return item


class WeightedTrainer(Trainer):
    """Cross-entropy berbobot untuk menangani class imbalance."""

    def __init__(self, *a, class_weights=None, **k):
        super().__init__(*a, **k)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(outputs.logits.device))
        loss = loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def main(args):
    ensure_dirs()
    df = pd.read_csv(args.data).dropna(subset=["sentimen"])
    df = df[df["judul_clean"].fillna("").str.len() > 0].reset_index(drop=True)
    labels = sorted(df["sentimen"].unique())
    l2i = {l: i for i, l in enumerate(labels)}
    y = df["sentimen"].map(l2i).values
    X = df["judul_clean"].tolist()

    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.2, stratify=y, random_state=SEED)
    idx_tr, idx_val = train_test_split(idx_tr, test_size=0.1, stratify=y[idx_tr], random_state=SEED)

    tok = AutoTokenizer.from_pretrained(args.model)
    enc = lambda ids: tok([X[i] for i in ids], truncation=True, padding="max_length", max_length=64)
    ds_tr, ds_val, ds_te = (DS(enc(i), y[i]) for i in (idx_tr, idx_val, idx_te))

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(labels), id2label=dict(enumerate(labels)), label2id=l2i)
    weights = torch.tensor(compute_class_weight("balanced", classes=np.arange(len(labels)), y=y[idx_tr]),
                           dtype=torch.float)

    def compute_metrics(p):
        pred = p.predictions.argmax(-1)
        return {"macro_f1": f1_score(p.label_ids, pred, average="macro")}

    targs = TrainingArguments(output_dir="outputs_private/indobert_runs", num_train_epochs=args.epochs,
                              per_device_train_batch_size=16, per_device_eval_batch_size=32,
                              learning_rate=2e-5, weight_decay=0.01, eval_strategy="epoch",
                              save_strategy="no", report_to="none", seed=SEED,
                              fp16=torch.cuda.is_available())
    trainer = WeightedTrainer(model=model, args=targs, train_dataset=ds_tr, eval_dataset=ds_val,
                              compute_metrics=compute_metrics, class_weights=weights)
    trainer.train()

    pred = trainer.predict(ds_te).predictions.argmax(-1)
    rep = classification_report(y[idx_te], pred, target_names=labels, output_dict=True, zero_division=0)
    print(classification_report(y[idx_te], pred, target_names=labels, zero_division=0))
    out = {"model": args.model, "test_macro_f1": round(f1_score(y[idx_te], pred, average="macro"), 4),
           "per_class": {l: {m: round(v, 4) for m, v in rep[l].items()} for l in labels}}
    (METRICS / "indobert_sentiment.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    trainer.save_model(str(MODELS / "indobert_sentiment"))
    tok.save_pretrained(str(MODELS / "indobert_sentiment"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA_PROCESSED / "clean.csv"))
    ap.add_argument("--model", default="indobenchmark/indobert-base-p1")
    ap.add_argument("--epochs", type=int, default=4)
    main(ap.parse_args())
