
import argparse
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import (GroupShuffleSplit, StratifiedGroupKFold, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from utils import DATA_PROCESSED, DEPLOY_MODELS, FIGURES, METRICS, MODELS, PRIVATE_OUT, ensure_dirs

SEED = 42
TASKS = {
    "sentiment": {"target": "sentimen", "group": None},
    "tone": {"target": "tone_grouped", "group": None},
    "topic": {"target": "topik", "group": None},
    "tier": {"target": "media_tier", "group": "sumber"},
}


def make_features(hashing=False):
    if hashing:
        # Mode deploy: HashingVectorizer TIDAK menyimpan kosakata, jadi file model tidak memuat
        # kata/nama dari data asli (hanya bobot per 'ember' hash).
        return FeatureUnion([
            ("word", Pipeline([
                ("hash", HashingVectorizer(ngram_range=(1, 2), n_features=2 ** 15, alternate_sign=False, norm=None)),
                ("tfidf", TfidfTransformer(sublinear_tf=True))])),
            ("char", Pipeline([
                ("hash", HashingVectorizer(analyzer="char_wb", ngram_range=(2, 5), n_features=2 ** 16,
                                           alternate_sign=False, norm=None)),
                ("tfidf", TfidfTransformer(sublinear_tf=True))])),
        ])
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
    ])


def candidates():
    return {
        "baseline_majority": DummyClassifier(strategy="most_frequent"),
        "logreg": LogisticRegression(max_iter=3000, C=5, class_weight="balanced"),
        "linear_svc": LinearSVC(C=0.5, class_weight="balanced"),
        "complement_nb": ComplementNB(alpha=0.3),
    }


def make_pipe(clf, hashing=False):
    return Pipeline([("features", make_features(hashing)), ("clf", clone(clf))])


def plot_confusion(y_true, y_pred, labels, path, title):
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    plt.figure(figsize=(1.2 * len(labels) + 3, 1.0 * len(labels) + 2.5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels, vmin=0, vmax=1)
    plt.xlabel("Prediksi")
    plt.ylabel("Label sebenarnya")
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_top_terms(pipe, path, k=15):
    clf = pipe.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return
    names = np.array(pipe.named_steps["features"].get_feature_names_out())
    rows = []
    coefs = clf.coef_
    classes = clf.classes_
    if coefs.shape[0] == 1:  # biner
        coefs, classes = np.vstack([-coefs[0], coefs[0]]), clf.classes_
    for cls, w in zip(classes, coefs):
        for i in np.argsort(w)[::-1][:k]:
            rows.append({"kelas": cls, "fitur": names[i], "bobot": round(float(w[i]), 3)})
    pd.DataFrame(rows).to_csv(path, index=False)


def main(args):
    ensure_dirs()
    cfg = TASKS[args.task]
    tcol, gcol = cfg["target"], cfg["group"]
    hashing = args.deploy
    tag_name = args.tag or ("deploy" if args.deploy else "")
    tag = f"_{tag_name}" if tag_name else ""

    df = pd.read_csv(args.data)
    text = df["judul_clean"].fillna("")
    if args.task == "tier" and args.use_domain:
        text = text + " " + df["domain"].fillna("").str.replace(".", " ", regex=False)
    df = df.assign(_text=text)
    df = df.dropna(subset=[tcol])
    df = df[df["_text"].str.len() > 0]

    vc = df[tcol].value_counts()
    small = vc[vc < args.min_class].index.tolist()
    if small:
        print(f"[info] kelas dibuang karena < {args.min_class} contoh: {small}")
        df = df[~df[tcol].isin(small)]
    df = df.reset_index(drop=True)
    X, y = df["_text"], df[tcol]
    groups = df[gcol] if gcol else None
    print(f"Tugas: {args.task} | n={len(df)} | kelas:\n{y.value_counts().to_string()}\n")

    # ---- split ----
    if gcol:
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(X, y, groups))
    else:
        tr, te = train_test_split(np.arange(len(df)), test_size=0.2, stratify=y, random_state=SEED)
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
    gtr = groups.iloc[tr] if gcol else None
    if gcol:
        print(f"Sumber di train: {gtr.nunique()} | sumber di test: {groups.iloc[te].nunique()} "
              f"| irisan: {len(set(gtr) & set(groups.iloc[te]))} (harus 0)\n")

    # ---- cross-validation di data latih ----
    n_splits = max(2, min(5, int(ytr.value_counts().min())))
    cv = (StratifiedGroupKFold(n_splits, shuffle=True, random_state=SEED) if gcol
          else StratifiedKFold(n_splits, shuffle=True, random_state=SEED))
    cv_res = {}
    for name, clf in candidates().items():
        s = cross_val_score(make_pipe(clf, hashing), Xtr, ytr, cv=cv, scoring="f1_macro", groups=gtr)
        cv_res[name] = {"mean": round(float(s.mean()), 4), "std": round(float(s.std()), 4)}
        print(f"CV macro-F1  {name:18s} {s.mean():.3f} ± {s.std():.3f}")
    best = max((k for k in cv_res if not k.startswith("baseline")), key=lambda k: cv_res[k]["mean"])
    print(f"\nModel terbaik (CV): {best}")

    # ---- evaluasi akhir di data uji (hanya sekali) ----
    base = make_pipe(candidates()["baseline_majority"], hashing).fit(Xtr, ytr)
    base_f1 = f1_score(yte, base.predict(Xte), average="macro", zero_division=0)
    pipe = make_pipe(candidates()[best], hashing).fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    labels = sorted(y.unique())
    report = classification_report(yte, pred, labels=labels, output_dict=True, zero_division=0)
    print("\n=== HASIL DATA UJI ===")
    print(classification_report(yte, pred, labels=labels, zero_division=0))
    print(f"Baseline (selalu tebak kelas mayoritas) macro-F1: {base_f1:.3f}")

    metrics = {
        "task": args.task, "data": str(args.data), "n_train": int(len(tr)), "n_test": int(len(te)),
        "classes": labels, "cv_macro_f1": cv_res, "best_model": best,
        "test": {
            "accuracy": round(accuracy_score(yte, pred), 4),
            "macro_f1": round(f1_score(yte, pred, average="macro", zero_division=0), 4),
            "weighted_f1": round(f1_score(yte, pred, average="weighted", zero_division=0), 4),
            "baseline_macro_f1": round(base_f1, 4),
            "per_class": {k: {m: round(v, 4) for m, v in report[k].items()} for k in labels},
        },
    }
    name = f"{args.task}{tag}"
    (METRICS / f"{name}.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    plot_confusion(yte, pred, labels, FIGURES / f"cm_{name}.png",
                   f"{args.task}: {best} (macro-F1 {metrics['test']['macro_f1']:.2f})")

    # file privat (berisi judul asli / kosakata) -> outputs_private/ (di-.gitignore)
    err = pd.DataFrame({"judul": df["judul"].iloc[te].values, "benar": yte.values, "prediksi": pred})
    err[err["benar"] != err["prediksi"]].to_csv(PRIVATE_OUT / f"errors_{name}.csv", index=False)
    if not hashing:  # fitur hashing tidak punya nama kata
        save_top_terms(pipe, PRIVATE_OUT / f"top_terms_{name}.csv")

    if hashing:
        joblib.dump(pipe, DEPLOY_MODELS / f"{args.task}.joblib", compress=3)
        import sklearn
        (DEPLOY_MODELS / "VERSIONS.txt").write_text(f"scikit-learn=={sklearn.__version__}\n")
        print(f"\nTersimpan: deploy_models/{args.task}.joblib (boleh di-push), reports/metrics/{name}.json")
    else:
        joblib.dump(pipe, MODELS / f"{name}.joblib")
        print(f"\nTersimpan: models/{name}.joblib, reports/metrics/{name}.json, reports/figures/cm_{name}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=TASKS, required=True)
    ap.add_argument("--data", default=str(DATA_PROCESSED / "clean.csv"))
    ap.add_argument("--min-class", type=int, default=10, help="buang kelas dengan contoh lebih sedikit")
    ap.add_argument("--use-domain", action="store_true",
                    help="(tier) tambahkan domain URL sebagai fitur; hati-hati: bisa membocorkan identitas sumber")
    ap.add_argument("--deploy", action="store_true",
                    help="latih model hashing tanpa kosakata -> deploy_models/ (aman untuk demo publik)")
    ap.add_argument("--tag", default="", help="akhiran nama file output, mis. 'sample'")
    main(ap.parse_args())
