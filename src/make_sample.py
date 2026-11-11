
import numpy as np
import pandas as pd

from prepare_data import add_derived
from utils import DATA_SAMPLE, ensure_dirs

rng = np.random.default_rng(42)
ORGS = ["Universitas Contoh Satu", "Universitas Contoh Dua", "Institut Contoh Tiga"]
CITIES = ["Kota Alfa", "Kota Beta", "Kota Gamma", "Kota Delta"]

# (template judul, kata kunci, tone) per sentimen
TEMPLATES = {
    "Positif": [
        ("{org} raih peringkat pertama pemeringkatan perguruan tinggi nasional", "{org}, peringkat, prestasi", "Dukungan"),
        ("Mahasiswa {org} juara lomba debat tingkat nasional", "mahasiswa, juara, lomba", "Inspiratif"),
        ("Dosen {org} publikasikan riset energi terbarukan di jurnal internasional", "riset, jurnal, publikasi", "Dukungan"),
        ("Beasiswa baru {org} bantu ratusan mahasiswa berprestasi", "beasiswa, mahasiswa", "Inspiratif"),
        ("{org} gelar bakti sosial dan program wakaf untuk warga sekitar", "wakaf, bakti sosial", "Dukungan"),
        ("Alumni {org} sukses bangun startup pendidikan digital", "alumni, startup", "Inspiratif"),
    ],
    "Negatif": [
        ("Sengketa aset yayasan dan {org} berlanjut ke jalur hukum", "sengketa, aset, yayasan", "Kritik"),
        ("Dugaan pemalsuan dokumen di lingkungan {org} dilaporkan ke polisi", "pemalsuan, pidana, polda", "Kritik"),
        ("Mahasiswa {org} protes kenaikan biaya kuliah, rektorat dikritik", "mahasiswa, protes", "Provokatif/Alarmis"),
        ("Kisruh pengelolaan lahan {org} memanas, warga menuntut kejelasan", "kisruh, lahan, konflik", "Kritik"),
        ("Polemik kebijakan kampus {org} picu kecaman publik", "polemik, kebijakan", "Provokatif/Alarmis"),
    ],
    "Netral": [
        ("{org} umumkan jadwal penerimaan mahasiswa baru tahun ajaran 2027", "mahasiswa baru, jadwal", "Informatif"),
        ("Rektor {org} hadiri rapat koordinasi perguruan tinggi", "rektor, rapat", "Netral"),
        ("Seminar nasional tentang kebijakan pendidikan digelar di {org}", "seminar, kebijakan", "Informatif"),
        ("{org} tandatangani kerja sama dengan pemerintah daerah", "kerja sama, pemda", "Netral"),
        ("Kuliah umum bahas masa depan pendidikan tinggi di {org}", "kuliah umum", "Netral"),
    ],
}
SOURCES = {  # sumber fiktif -> (tier, kredibilitas)
    "Harian Contoh Nasional": ("Tier 1: Internasional/Nasional", "Tinggi"),
    "Kabar Contoh Indonesia": ("Tier 1: Internasional/Nasional", "Tinggi"),
    "Portal Contoh News": ("Tier 1: Internasional/Nasional", "Tinggi"),
    "Warta Contoh Republik": ("Tier 1: Internasional/Nasional", "Tinggi"),
    "Suara Contoh Regional": ("Tier 2: Regional", "Tinggi"),
    "Kabar Contoh Daerah": ("Tier 2: Regional", "Sedang"),
    "Berita Contoh Wilayah": ("Tier 2: Regional", "Sedang"),
    "Radar Contoh Raya": ("Tier 2: Regional", "Sedang"),
    "Buletin Contoh Kampus": ("Tier 3: Lokal/Niche", "Sedang"),
    "Blog Contoh Komunitas": ("Tier 3: Lokal/Niche", "Rendah"),
    "Majalah Contoh Lokal": ("Tier 3: Lokal/Niche", "Sedang"),
    "Info Contoh Warga": ("Tier 3: Lokal/Niche", "Rendah"),
}


def main(n=150):
    rows = []
    labels = rng.choice(list(TEMPLATES), size=n, p=[0.5, 0.25, 0.25])
    src_names = list(SOURCES)
    for i, sent in enumerate(labels):
        tpl, kw, tone = TEMPLATES[sent][rng.integers(len(TEMPLATES[sent]))]
        org = ORGS[rng.integers(len(ORGS))]
        judul = tpl.format(org=org)
        if rng.random() < 0.5:
            judul += f" di {CITIES[rng.integers(len(CITIES))]}"
        src = src_names[rng.integers(len(src_names))]
        tier, cred = SOURCES[src]
        rows.append({
            "tanggal": pd.Timestamp("2026-01-01") + pd.Timedelta(days=int(rng.integers(0, 300))),
            "pic": f"PIC-{rng.integers(1, 4)}", "media": "portal berita online", "sumber": src,
            "judul": judul, "media_tier": tier, "kredibilitas": cred, "link": np.nan,
            "kata_kunci": kw.format(org=org), "sentimen": sent, "tone": tone,
            "jenis_tokoh": rng.choice(["Kolektif", "Institut", "Individual"]),
        })
    df = add_derived(pd.DataFrame(rows))
    ensure_dirs()
    out = DATA_SAMPLE / "sample_synthetic.csv"
    df.to_csv(out, index=False)
    print(f"{len(df)} baris sintetis -> {out}")
    print(df["sentimen"].value_counts().to_string())


if __name__ == "__main__":
    main()
