
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
import dashboard as dash  # noqa: E402
from utils import DEPLOY_MODELS, MODELS, predict_with_scores  # noqa: E402

st.set_page_config(page_title="Media Monitoring ML", page_icon="📰", layout="wide")
st.title("📰 Media Monitoring ML")
st.caption("Prediksi sentimen, tone, dan topik berita + ringkasan hasil monitoring media.")

TASKS = {"sentiment": "Sentimen", "tone": "Tone", "topic": "Topik"}
DASH_DATA = ROOT / "reports" / "dashboard_data.json"


@st.cache_resource
def load_model(name):
    # Di server (deploy) hanya ada deploy_models/. Di komputer lokal, model penuh di models/ dipakai bila ada.
    for p in (MODELS / f"{name}.joblib", DEPLOY_MODELS / f"{name}.joblib"):
        if p.exists():
            return joblib.load(p)
    return None


@st.cache_data
def load_dashboard(path):
    return dash.load(path)


def show(fig):
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


tab_pred, tab_dash = st.tabs(["🔮 Prediksi", "📊 Dashboard"])

# ------------------------------------------------------------------ TAB PREDIKSI
with tab_pred:
    models = {k: load_model(k) for k in TASKS}
    if not any(models.values()):
        st.error("Model belum ada. Jalankan: python src/train.py --task sentiment --deploy (juga tone dan topic)")
    else:
        judul = st.text_area("Judul berita", height=100, placeholder="Tempel judul berita di sini...")
        if st.button("Prediksi", type="primary") and judul.strip():
            cols = st.columns(len(TASKS))
            for col, (key, label) in zip(cols, TASKS.items()):
                if models[key] is None:
                    col.warning(f"Model {label} belum dilatih")
                    continue
                scores = predict_with_scores(models[key], judul)
                col.metric(label, scores[0][0], f"{scores[0][1]:.0%}")
                col.dataframe(pd.DataFrame(scores, columns=["kelas", "skor"]).set_index("kelas")
                              .style.format("{:.0%}"))
            st.caption("Skor = probabilitas (LogReg/NB) atau softmax dari skor SVM, jadi anggap sebagai "
                       "tingkat keyakinan relatif.")

    with st.expander("Tentang model ini"):
        st.markdown(
            "- Model: TF-IDF (kata + karakter, di-hash) + model linear klasik, dilatih pada ±830 judul berita "
            "monitoring media sebuah institusi pendidikan tinggi.\n"
            "- Model terlatih tidak menyimpan kosakata/judul asli; data asli tidak dipublikasikan.\n"
            "- Keterbatasan: data kecil, satu institusi, kelas *Netral* paling sulit dikenali. "
            "Gunakan sebagai demo, bukan alat keputusan.")

# ------------------------------------------------------------------ TAB DASHBOARD
with tab_dash:
    if not DASH_DATA.exists():
        st.info("Data dashboard belum dibuat. Jalankan: python src/make_dashboard_data.py")
    else:
        cube, sumber, meta = load_dashboard(str(DASH_DATA))
        st.caption(f"{meta['n_total']} berita • {meta['periode_awal']} s.d. {meta['periode_akhir']} • "
                   f"{meta['n_sumber']} media. Dashboard hanya memakai angka agregat (tanpa judul/nama).")

        c1, c2 = st.columns(2)
        tiers_all = sorted(cube["media_tier"].unique())
        topik_all = sorted(cube["topik"].unique())
        tiers = c1.multiselect("Filter tier media", tiers_all, default=tiers_all)
        topiks = c2.multiselect("Filter topik", topik_all, default=topik_all)
        f = cube[cube["media_tier"].isin(tiers) & cube["topik"].isin(topiks)]

        k = dash.kpis(f)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total berita", f"{k['total']:,}")
        m2.metric("Positif", f"{k['positif']:.0f}%")
        m3.metric("Netral", f"{k['netral']:.0f}%")
        m4.metric("Negatif", f"{k['negatif']:.0f}%")

        a, b = st.columns(2)
        with a:
            show(dash.fig_sentimen(f))
        with b:
            show(dash.fig_tren(f))

        st.subheader("Prediksi jumlah berita media UIN Jakarta")
        horizon = st.slider("Prediksi untuk berapa bulan ke depan?", 1, 6, 3)
        forecast, basis = dash.forecast_monthly(
            f, periods=horizon, observed_through=meta["periode_akhir"]
        )
        if forecast.empty:
            st.info("Prediksi memerlukan setidaknya tiga bulan kalender lengkap pada filter ini.")
        else:
            show(dash.fig_forecast(f, forecast))
            st.dataframe(
                forecast.rename(columns={"bulan": "Bulan", "prediksi": "Perkiraan berita"}),
                hide_index=True,
                use_container_width=True,
            )
            st.caption(
                f"Baseline = rata-rata tiga bulan lengkap terakhir ({basis[0]}–{basis[-1]}). "
                "Prediksi mengikuti filter tier dan topik di atas."
            )
            st.caption(
                "Ini baseline sederhana, bukan model musiman: riwayat baru 10 bulan dan "
                "bulan terakhir yang belum lengkap tidak dipakai untuk menghitung baseline."
            )

        a, b = st.columns(2)
        with a:
            show(dash.fig_tier(f))
        with b:
            show(dash.fig_topik(f))
        a, b = st.columns(2)
        with a:
            show(dash.fig_tone(f))
        with b:
            show(dash.fig_sumber(sumber))
            st.caption("Grafik media tidak ikut terfilter.")