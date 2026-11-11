
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SENT_ORDER = ["Positif", "Netral", "Negatif"]
COLORS = {"Positif": "#2E8B57", "Netral": "#9AA0A6", "Negatif": "#C0392B"}
BLUE = "#3B6EA5"

plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "axes.axisbelow": True, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left",
})


def load(path):
    d = json.loads(Path(path).read_text())
    return pd.DataFrame(d["cube"]), pd.DataFrame(d["sumber"]), d["meta"]


def _pivot(df, index, order=None):
    p = df.pivot_table(index=index, columns="sentimen", values="n", aggfunc="sum", fill_value=0)
    for s in SENT_ORDER:
        if s not in p:
            p[s] = 0
    p = p[SENT_ORDER]
    return p.reindex(order) if order is not None else p


def _empty(msg="Tidak ada data untuk filter ini"):
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.text(0.5, 0.5, msg, ha="center", va="center", color="#777")
    ax.axis("off")
    return fig


def kpis(cube):
    total = int(cube["n"].sum())
    by = cube.groupby("sentimen")["n"].sum()
    pct = lambda s: (by.get(s, 0) / total * 100) if total else 0
    return {"total": total, "positif": pct("Positif"), "netral": pct("Netral"), "negatif": pct("Negatif")}


def fig_sentimen(cube):
    if cube.empty:
        return _empty()
    s = cube.groupby("sentimen")["n"].sum().reindex(SENT_ORDER).fillna(0)
    fig, ax = plt.subplots(figsize=(6, 2.8))
    bars = ax.barh(s.index[::-1], s.values[::-1], color=[COLORS[i] for i in s.index[::-1]])
    ax.bar_label(bars, padding=3)
    ax.set_title("Sebaran sentimen")
    ax.set_xlabel("Jumlah berita")
    ax.grid(axis="y", visible=False)
    return fig


def fig_tren(cube):
    d = cube[cube["bulan"] != "Tidak diketahui"]
    if d.empty:
        return _empty()
    p = _pivot(d, "bulan").sort_index()
    fig, ax = plt.subplots(figsize=(8, 3.4))
    bottom = pd.Series(0, index=p.index)
    for s in SENT_ORDER:
        ax.bar(p.index, p[s], bottom=bottom, color=COLORS[s], label=s)
        bottom += p[s]
    ax.set_title("Jumlah berita per bulan")
    ax.set_ylabel("Berita")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.grid(axis="x", visible=False)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    return fig


def forecast_monthly(cube, periods=3, observed_through=None):
    """Proyeksi baseline volume berita dari rata-rata tiga bulan lengkap terakhir."""
    if periods < 1:
        raise ValueError("periods harus minimal 1")

    empty = pd.DataFrame(columns=["bulan", "prediksi"])
    d = cube[cube["bulan"] != "Tidak diketahui"].copy()
    d["periode"] = pd.to_datetime(d["bulan"], format="%Y-%m", errors="coerce").dt.to_period("M")
    d = d.dropna(subset=["periode"])
    if d.empty:
        return empty, []

    observed_period = d["periode"].max()
    if observed_through is not None:
        observed_date = pd.to_datetime(observed_through, errors="coerce")
        if pd.isna(observed_date):
            raise ValueError("observed_through harus berupa tanggal yang valid")
        observed_period = observed_date.to_period("M")
        last_complete = observed_period if observed_date.is_month_end else observed_period - 1
    else:
        last_complete = observed_period

    first_period = d["periode"].min()
    if last_complete < first_period:
        return empty, []

    history = d.groupby("periode")["n"].sum()
    complete_months = pd.period_range(first_period, last_complete, freq="M")
    history = history.reindex(complete_months, fill_value=0)
    if len(history) < 3:
        return empty, []

    basis = history.tail(3)
    estimate = int(round(float(basis.mean())))
    future_months = pd.period_range(observed_period + 1, periods=periods, freq="M")
    forecast = pd.DataFrame({
        "bulan": future_months.astype(str),
        "prediksi": estimate,
    })
    return forecast, basis.index.astype(str).tolist()


def fig_forecast(cube, forecast):
    d = cube[cube["bulan"] != "Tidak diketahui"].copy()
    d["periode"] = pd.to_datetime(d["bulan"], format="%Y-%m", errors="coerce").dt.to_period("M")
    d = d.dropna(subset=["periode"])
    if d.empty or forecast.empty:
        return _empty()

    actual = d.groupby("periode")["n"].sum()
    periods = pd.period_range(actual.index.min(), actual.index.max(), freq="M")
    actual = actual.reindex(periods, fill_value=0)
    labels = periods.astype(str).tolist() + forecast["bulan"].tolist()
    actual_x = range(len(actual))
    forecast_x = range(len(actual) - 1, len(labels))

    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(actual_x, actual.values, marker="o", color=BLUE, label="Aktual")
    ax.plot(
        forecast_x,
        [forecast["prediksi"].iloc[0], *forecast["prediksi"].tolist()],
        marker="o",
        linestyle="--",
        color="#E67E22",
        label="Proyeksi baseline",
    )
    ax.axvline(len(actual) - 0.5, color="#999", linestyle=":", linewidth=1)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_title("Proyeksi jumlah berita media UIN Jakarta")
    ax.set_ylabel("Berita")
    ax.legend(frameon=False)
    ax.grid(axis="x", visible=False)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    return fig


def _share_bar(cube, col, title, order=None):
    d = cube[cube[col] != "Tidak diketahui"]
    if d.empty:
        return _empty()
    p = _pivot(d, col, order)
    p = p.loc[p.sum(axis=1) > 0]
    share = p.div(p.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(p) + 1.6))
    left = pd.Series(0.0, index=share.index)
    for s in SENT_ORDER:
        ax.barh(share.index, share[s], left=left, color=COLORS[s], label=s)
        for y, (v, l) in enumerate(zip(share[s], left)):
            if v >= 8:
                ax.text(l + v / 2, y, f"{v:.0f}%", ha="center", va="center", color="white", fontsize=9)
        left += share[s]
    for y, n in enumerate(p.sum(axis=1)):
        ax.text(101, y, f"n={int(n)}", va="center", fontsize=9, color="#555")
    ax.set_xlim(0, 112)
    ax.set_xlabel("% berita")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.45, -0.35 - 0.02 * len(p)))
    ax.grid(axis="y", visible=False)
    return fig


def fig_tier(cube):
    return _share_bar(cube, "media_tier", "Proporsi sentimen per tier media", sorted(cube["media_tier"].unique()))


def fig_topik(cube):
    order = cube.groupby("topik")["n"].sum().sort_values(ascending=False).index.tolist() if not cube.empty else None
    return _share_bar(cube, "topik", "Proporsi sentimen per topik", order)


def fig_tone(cube):
    d = cube[cube["tone"] != "Tidak diketahui"]
    if d.empty:
        return _empty()
    ct = d.pivot_table(index="tone", columns="sentimen", values="n", aggfunc="sum", fill_value=0)
    ct = ct.reindex(columns=SENT_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 0.5 * len(ct) + 1.6))
    im = ax.imshow(ct.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(ct.columns)), ct.columns)
    ax.set_yticks(range(len(ct.index)), ct.index)
    thr = ct.values.max() / 2 if ct.values.max() else 0
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            v = ct.values[i, j]
            ax.text(j, i, int(v), ha="center", va="center", color="white" if v > thr else "#222")
    ax.set_title("Tone vs sentimen (jumlah berita)")
    ax.grid(False)
    return fig


def fig_sumber(sumber):
    if sumber.empty:
        return _empty()
    p = _pivot(sumber, "sumber")
    p = p.loc[p.sum(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(p) + 1.4))
    left = pd.Series(0, index=p.index)
    for s in SENT_ORDER:
        ax.barh(p.index, p[s], left=left, color=COLORS[s], label=s)
        left += p[s]
    ax.set_title(f"{len(p)} media dengan berita terbanyak")
    ax.set_xlabel("Jumlah berita")
    ax.legend(frameon=False, ncol=3, loc="lower right")
    ax.grid(axis="y", visible=False)
    return fig