"""Membuat data AGREGAT untuk dashboard (aman dipublikasikan).

Hanya berisi jumlah berita per kombinasi bulan/tier/topik/tone/sentimen, dan jumlah per nama media
(top 12). TIDAK ada judul, link, nama tokoh, atau PIC.

    python src/make_dashboard_data.py
Output: reports/dashboard_data.json
"""
import json

import pandas as pd

from utils import DATA_PROCESSED, ROOT, ensure_dirs

TOP_SUMBER = 12
OUT = ROOT / "reports" / "dashboard_data.json"


def main(path=DATA_PROCESSED / "clean.csv"):
    ensure_dirs()
    df = pd.read_csv(path, parse_dates=["tanggal"])
    df["bulan"] = df["tanggal"].dt.strftime("%Y-%m")
    for c in ["bulan", "media_tier", "topik", "tone_grouped"]:
        df[c] = df[c].fillna("Tidak diketahui")

    cube = (df.groupby(["bulan", "media_tier", "topik", "tone_grouped", "sentimen"]).size()
              .reset_index(name="n").rename(columns={"tone_grouped": "tone"}))

    top = df["sumber"].value_counts().head(TOP_SUMBER).index
    sumber = (df[df["sumber"].isin(top)].groupby(["sumber", "sentimen"]).size().reset_index(name="n"))

    meta = {"n_total": int(len(df)), "periode_awal": df["tanggal"].min().strftime("%Y-%m-%d"),
            "periode_akhir": df["tanggal"].max().strftime("%Y-%m-%d"), "n_sumber": int(df["sumber"].nunique())}
    OUT.write_text(json.dumps({"meta": meta, "cube": cube.to_dict("records"),
                               "sumber": sumber.to_dict("records")}, ensure_ascii=False))
    print(f"Tersimpan: {OUT} | {len(cube)} baris agregat | {meta}")


if __name__ == "__main__":
    main()