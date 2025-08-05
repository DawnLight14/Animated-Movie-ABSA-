# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from absa_pipeline import run_absa_pipeline

st.set_page_config(page_title="ABSA Film Animasi", layout="wide")

st.title("🎬 Sistem Analisis Sentimen Berbasis Aspek (ABSA)")

st.markdown("""
Masukkan ulasan film animasi dalam **bahasa Inggris**.  
Sistem akan mendeteksi **aspek** dan memberikan **sentimen** untuk tiap aspek.
""")

# Inisialisasi session_state
if "all_results" not in st.session_state:
    st.session_state["all_results"] = pd.DataFrame(columns=["sentence", "aspect", "sentiment"])

review_text = st.text_area("Masukkan Ulasan Film:", height=200,
                           placeholder="Contoh: The animation was fantastic but the story was boring.")

if st.button("Analisis Sentimen"):
    if review_text.strip() == "":
        st.warning("⚠️ Masukkan teks ulasan terlebih dahulu.")
    else:
        with st.spinner("⏳ Memproses ulasan..."):
            results = run_absa_pipeline(review_text)
            df_new = pd.DataFrame(results)

        if df_new.empty:
            st.info("Tidak ada aspek ditemukan.")
        else:
            st.success("✅ Analisis selesai!")

            # Tambahkan hasil baru ke session_state
            st.session_state["all_results"] = pd.concat(
                [st.session_state["all_results"], df_new], ignore_index=True
            )

# Tampilkan hasil kumulatif
df_results = st.session_state["all_results"]

if not df_results.empty:
    st.subheader("📊 Hasil Analisis")
    st.dataframe(df_results)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Distribusi Sentimen")
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.countplot(x="sentiment", data=df_results, ax=ax, palette="coolwarm")
        st.pyplot(fig)

    with col2:
        st.subheader("📊 Distribusi Aspek")
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        df_results["aspect"].value_counts().plot(kind="bar", ax=ax2, color="skyblue")
        st.pyplot(fig2)

# Tombol reset jika ingin menghapus semua hasil
with st.expander("🧹 Reset Semua Hasil"):
    if st.button("🔁 Hapus Hasil Analisis"):
        st.session_state["all_results"] = pd.DataFrame(columns=["sentence", "aspect", "sentiment"])
        st.success("Hasil telah direset.")
