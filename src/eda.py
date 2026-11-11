
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import DATA_PROCESSED, FIGURES, ensure_dirs

ensure_dirs()
sns.set_theme(style="whitegrid")
df = pd.read_csv(DATA_PROCESSED / "clean.csv", parse_dates=["tanggal"])
print(df.shape)


def save(name):
    plt.tight_layout()
    plt.savefig(FIGURES / name, dpi=150)
    plt.close()


# %% 1. Sebaran sentimen (menunjukkan class imbalance)
order = df["sentimen"].value_counts().index
ax = sns.countplot(data=df, y="sentimen", order=order, color="#4C72B0")
for c in ax.containers:
    ax.bar_label(c, padding=3)
ax.set_title("Sebaran sentimen berita")
save("eda_sentimen.png")

# %% 2. Sentimen x Tone
ct = pd.crosstab(df["sentimen"], df["tone_grouped"])
sns.heatmap(ct, annot=True, fmt="d", cmap="Blues")
plt.title("Sentimen vs Tone (jumlah berita)")
save("eda_sentimen_tone.png")

# %% 3. Tren bulanan
m = df.dropna(subset=["tanggal"]).assign(bulan=lambda d: d["tanggal"].dt.to_period("M").astype(str))
tren = m.groupby(["bulan", "sentimen"]).size().unstack(fill_value=0)
tren.plot(kind="bar", stacked=True, figsize=(9, 4), color=["#C44E52", "#8C8C8C", "#55A868"][: tren.shape[1]])
plt.title("Jumlah berita per bulan menurut sentimen")
plt.xlabel("")
save("eda_tren_bulanan.png")

# %% 4. Tier media dan sentimen
ct = pd.crosstab(df["media_tier"], df["sentimen"], normalize="index")
ct.plot(kind="barh", stacked=True, figsize=(8, 3.5), color=["#C44E52", "#8C8C8C", "#55A868"][: ct.shape[1]])
plt.title("Proporsi sentimen per tier media")
save("eda_tier_sentimen.png")

# %% 5. Sumber teratas
top = df["sumber"].value_counts().head(15)
sns.barplot(x=top.values, y=top.index, color="#4C72B0")
plt.title("15 sumber berita teratas")
save("eda_top_sumber.png")

# %% 6. Topik
t = df["topik"].value_counts()
sns.barplot(x=t.values, y=t.index, color="#4C72B0")
plt.title("Sebaran topik (dari aturan kata kunci)")
save("eda_topik.png")

# %% 7. Panjang judul
df["panjang_judul"] = df["judul"].str.len()
sns.histplot(df["panjang_judul"], bins=30)
plt.title("Panjang judul (karakter)")
save("eda_panjang_judul.png")

# %% 8. Konsistensi antar-PIC: distribusi sentimen per PIC (indikasi bias labeler)
ct = pd.crosstab(df["pic"], df["sentimen"], normalize="index").round(2)
print("\nProporsi sentimen per PIC:\n", ct)
print("\nGambar tersimpan di", FIGURES)
