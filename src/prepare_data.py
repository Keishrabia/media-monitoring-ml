
import argparse
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from utils import DATA_PROCESSED, DATA_RAW, clean_text, ensure_dirs

# nama kolom asli (setelah di-strip) -> nama kolom kerja
# Kolom 'Tokoh yang disebut', 'Catatan Analyst', dan kolom engagement sengaja TIDAK dibawa
# (berisi nama orang / hampir kosong).
COLUMNS = {
    "Tanggal Berita Dipublikasikan": "tanggal",
    "PIC": "pic",
    "Media": "media",
    "Sumber": "sumber",
    "Judul Berita": "judul",
    "Media Tier": "media_tier",
    "Kredibilitas": "kredibilitas",
    "Link/Source": "link",
    "Kata Kunci": "kata_kunci",
    "Sentimen": "sentimen",
    "Tone": "tone",
    "Jenis Tokoh": "jenis_tokoh",
}

# Aturan topik (weak labeling dari kolom Kata Kunci). Urutan penting: aturan pertama yang cocok menang.
# Silakan sesuaikan setelah melihat kata kunci di datamu sendiri.
TOPIC_RULES = [
    ("hukum_sengketa", ["sengketa", "pidana", "polda", "laporkan", "pemalsuan", "yayasan", "pamulang",
                        "konflik", "polemik", "kisruh", "korupsi", "legal", "bmn", "aset", "gugat",
                        "kuasa hukum"]),
    ("prestasi_peringkat", ["peringkat", "prestasi", "penghargaan", "juara", "ranking", "sinta",
                            "akreditasi", "award", "terbaik", "medali"]),
    ("riset_publikasi", ["riset", "penelitian", "publikasi", "jurnal", "karya ilmiah", "scopus",
                         "guru besar", "inovasi", "dosen"]),
    ("kerja_sama_kemitraan", ["kerja sama", "kerjasama", "kolaborasi", "gandeng", "mou", "kemitraan",
                              "bertemu", "pertemuan", "mitra"]),
    ("kegiatan_mahasiswa", ["mahasiswa", "aiesec", "pbak", "kkn", "organisasi", "youth", "exchange",
                            "beasiswa", "wisuda", "alumni", "kompetisi", "lomba"]),
    ("keagamaan_sosial", ["wakaf", "tabayun", "ramadan", "ramadhan", "zakat", "dakwah", "islam",
                          "haji", "halal", "masjid", "sosial", "bakti"]),
    ("politik_isu_publik", ["pilkada", "politik", "pemilu", "kebijakan", "demokrasi", "presiden",
                            "pemerintah", "dprd", "isu"]),
]

TONE_MERGE = {"Alarmis": "Provokatif/Alarmis", "Provokatif": "Provokatif/Alarmis"}


def assign_topic(kata_kunci):
    if pd.isna(kata_kunci) or not str(kata_kunci).strip():
        return np.nan
    k = str(kata_kunci).lower()
    for name, patterns in TOPIC_RULES:
        if any(p in k for p in patterns):
            return name
    return "lainnya"


def extract_domain(link):
    if pd.isna(link) or not str(link).lower().startswith("http"):
        return np.nan
    host = urlparse(str(link)).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Kolom turunan. Dipakai juga oleh make_sample.py agar skemanya identik."""
    df = df.copy()
    df["judul_clean"] = df["judul"].map(clean_text)
    df["tone_grouped"] = df["tone"].replace(TONE_MERGE)
    df["topik"] = df["kata_kunci"].map(assign_topic)
    df["domain"] = df["link"].map(extract_domain)
    return df


def main(xlsx_path):
    ensure_dirs()
    df = pd.read_excel(xlsx_path, sheet_name="Master Data")
    df.columns = df.columns.str.strip()
    df = df[list(COLUMNS)].rename(columns=COLUMNS)
    n0 = len(df)

    # rapikan string
    for c in df.select_dtypes(include="object"):
        df[c] = df[c].astype("string").str.strip().replace("", pd.NA)

    df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")
    df["sentimen"] = df["sentimen"].str.capitalize()
    df["tone"] = df["tone"].str.capitalize()
    df["jenis_tokoh"] = df["jenis_tokoh"].replace({"Koletif": "Kolektif"})
    df["media"] = df["media"].str.lower()
    df["sumber"] = df["sumber"].str.strip()

    # buang baris tanpa judul atau tanpa label sentimen, lalu duplikat judul
    df = df.dropna(subset=["judul", "sentimen"])
    n1 = len(df)
    df = df.drop_duplicates(subset=["judul"], keep="first")
    n2 = len(df)

    df = add_derived(df).reset_index(drop=True)
    out = DATA_PROCESSED / "clean.csv"
    df.to_csv(out, index=False)

    print(f"Baris awal           : {n0}")
    print(f"Setelah buang kosong : {n1}")
    print(f"Setelah buang duplikat: {n2}")
    print(f"Disimpan ke          : {out}\n")
    for c in ["sentimen", "tone_grouped", "topik", "media_tier"]:
        print(df[c].value_counts(dropna=False).to_string(), "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default=str(DATA_RAW / "media_monitoring.xlsx"))
    main(ap.parse_args().xlsx)
