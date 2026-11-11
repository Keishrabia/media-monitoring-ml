
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_SAMPLE = ROOT / "data" / "sample"
MODELS = ROOT / "models"  # model lokal (berisi kosakata asli) -> di-.gitignore
DEPLOY_MODELS = ROOT / "deploy_models"  # model hashing tanpa kosakata -> boleh di-push untuk demo
FIGURES = ROOT / "reports" / "figures"
METRICS = ROOT / "reports" / "metrics"
PRIVATE_OUT = ROOT / "outputs_private"  # di-.gitignore: berisi judul asli / kosakata


def ensure_dirs():
    for p in (DATA_RAW, DATA_PROCESSED, DATA_SAMPLE, MODELS, DEPLOY_MODELS, FIGURES, METRICS, PRIVATE_OUT):
        p.mkdir(parents=True, exist_ok=True)


_URL = re.compile(r"https?://\S+|www\.\S+")
_NON_ALNUM = re.compile(r"[^a-z0-9\s]")


def clean_text(s) -> str:
    """Lowercase, hapus URL & simbol, rapikan spasi. Dipakai saat training DAN saat prediksi."""
    s = str(s).lower()
    s = _URL.sub(" ", s)
    s = _NON_ALNUM.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def _softmax(z):
    z = np.asarray(z, dtype=float)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def predict_with_scores(pipe, text):
    """Kembalikan [(kelas, skor), ...] urut dari yang tertinggi.
    Pakai predict_proba bila ada, kalau tidak (LinearSVC) pakai softmax dari decision_function."""
    x = [clean_text(text)]
    classes = pipe.classes_
    if hasattr(pipe, "predict_proba"):
        p = pipe.predict_proba(x)[0]
    else:
        z = np.atleast_1d(pipe.decision_function(x)[0])
        p = _softmax(z) if len(z) == len(classes) else np.array([1 - _softmax([0, z[0]])[1], _softmax([0, z[0]])[1]])
    order = np.argsort(p)[::-1]
    return [(classes[i], float(p[i])) for i in order]
