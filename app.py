import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Konfigurasi Halaman Web
st.set_page_config(
    page_title="Spotify Analytics: Unsupervised & Supervised",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 1. FUNGSI MEMUAT DATA & MODEL (CACHING)
# ==========================================
@st.cache_resource
def load_assets():
    try:
        df = pd.read_csv('spotify_deployed.csv')

        with open('scaler_spotify.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('kmeans_spotify.pkl', 'rb') as f:
            kmeans = pickle.load(f)
        with open('naivebayes_spotify.pkl', 'rb') as f:
            gnb = pickle.load(f)
        with open('logreg_spotify.pkl', 'rb') as f:
            logreg = pickle.load(f)

        return df, scaler, kmeans, gnb, logreg
    except FileNotFoundError:
        st.error("File Model (.pkl) belum lengkap. Pastikan Anda sudah mengekspor StandardScaler, KMeans, NaiveBayes, dan LogisticRegression dari notebook.")
        return None, None, None, None, None

df, scaler, kmeans, gnb, logreg = load_assets()

# ==========================================
# 2. NAVIGASI SIDEBAR
# ==========================================
st.sidebar.title("Spotify Analytics System")
st.sidebar.markdown("---")
pilihan_halaman = st.sidebar.radio(
    "Pilih Modul Sistem:",
    ["Beranda dan Pemahaman Bisnis",
     "Prediksi Mood dan Auto-Playlist (Naive Bayes)",
     "Moderasi Konten Eksplisit (LogReg)"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Sistem ini dibangun menggunakan:")
st.sidebar.caption("1. Unsupervised: K-Means")
st.sidebar.caption("2. Supervised: Naive Bayes & Logistic Regression")

# ==========================================
# 3. HALAMAN 1: BERANDA
# ==========================================
if pilihan_halaman == "Beranda dan Pemahaman Bisnis":
    st.title("Sistem Analisis dan Prediksi Karakteristik Audio Spotify")
    st.write("Aplikasi ini mendemonstrasikan integrasi antara metodologi Unsupervised Learning untuk membuat segmentasi pasar, dan Supervised Learning untuk melakukan klasifikasi lagu baru.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Fase 1: Labeling (K-Means)")
        st.info(
            "Menggunakan K-Means Clustering, kami mengelompokkan 114.000 lagu tanpa label emosi "
            "menjadi dua segmen (klaster) utama secara objektif berdasarkan fitur akustiknya:"
        )
        st.write("- Cluster 0 (Relaksasi dan Fokus): Tempo lambat, akustik tinggi, energi rendah.")
        st.write("- Cluster 1 (Agresif dan Berenergi): Tempo kencang, distorsi, energi tinggi.")

    with col2:
        st.subheader("Fase 2: Klasifikasi (Supervised)")
        st.success(
            "Hasil pelabelan K-Means tersebut kami pelajari menggunakan algoritma klasifikasi probabilistik "
            "untuk memprediksi lagu baru di masa depan:"
        )
        st.write("- Naive Bayes: Digunakan untuk memprediksi Klaster (Mood) dari lagu baru.")
        st.write("- Logistic Regression: Digunakan untuk mendeteksi apakah lagu baru mengandung unsur eksplisit atau kasar.")

# ==========================================
# 4. HALAMAN 2: PREDIKSI MOOD & PLAYLIST (NAIVE BAYES)
# ==========================================
elif pilihan_halaman == "Prediksi Mood dan Auto-Playlist (Naive Bayes)":
    st.title("Smart Auto-Playlist berbasis Naive Bayes")
    st.write("Uji model Naive Bayes Classifier kami. Masukkan metrik audio dari sebuah trek baru, dan sistem akan memprediksi nuansa emosinya serta mencarikan 5 lagu referensi.")

    st.markdown("### Masukkan Fitur Audio Lagu Baru:")
    col1, col2, col3 = st.columns(3)

    with col1:
        danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
        energy = st.slider("Energy", 0.0, 1.0, 0.5)
        loudness = st.slider("Loudness (dB)", -60.0, 5.0, -10.0)
    with col2:
        speechiness = st.slider("Speechiness", 0.0, 1.0, 0.1)
        acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5)
        instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0)
    with col3:
        liveness = st.slider("Liveness", 0.0, 1.0, 0.1)
        valence = st.slider("Valence (Keceriaan)", 0.0, 1.0, 0.5)
        tempo = st.slider("Tempo (BPM)", 0.0, 250.0, 120.0)

    if st.button("Prediksi Mood dan Rekomendasikan Lagu", type="primary"):
        if gnb and scaler:
            input_data = pd.DataFrame([[danceability, energy, loudness, speechiness,
                                        acousticness, instrumentalness, liveness, valence, tempo]],
                                      columns=['danceability', 'energy', 'loudness', 'speechiness',
                                               'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'])

            input_scaled = scaler.transform(input_data)
            hasil_prediksi = gnb.predict(input_scaled)[0]
            probabilitas = gnb.predict_proba(input_scaled)[0]

            st.markdown("---")
            st.subheader("Hasil Prediksi Naive Bayes")

            col_hasil, col_rekom = st.columns([1, 2])

            with col_hasil:
                if hasil_prediksi == 0:
                    st.success("Segmen Terprediksi: Cluster 0 (Relaksasi)")
                    st.metric("Tingkat Keyakinan (Probabilitas)", f"{probabilitas[0]*100:.2f}%")
                else:
                    st.error("Segmen Terprediksi: Cluster 1 (Agresif)")
                    st.metric("Tingkat Keyakinan (Probabilitas)", f"{probabilitas[1]*100:.2f}%")

            with col_rekom:
                st.write("**5 Lagu Serupa di Database Kami:**")
                df_rekomendasi = df[df['cluster'] == hasil_prediksi].sample(5)[['track_name', 'artists', 'track_genre']]
                df_rekomendasi.columns = ['Judul Lagu', 'Artis', 'Genre']
                st.dataframe(df_rekomendasi, use_container_width=True)

# ==========================================
# 5. HALAMAN 3: MODERASI KONTEN (LOGISTIC REGRESSION)
# ==========================================
elif pilihan_halaman == "Moderasi Konten Eksplisit (LogReg)":
    st.title("Sistem Moderasi Keamanan Konten (Logistic Regression)")
    st.write("Platform musik digital memerlukan pendeteksian dini terhadap konten dewasa. Model Logistic Regression kami dapat mendeteksi probabilitas lagu memiliki lirik eksplisit murni berdasarkan komposisi instrumen dan gaya audionya.")

    st.markdown("### Masukkan Fitur Audio:")
    col1, col2 = st.columns(2)

    with col1:
        speechiness = st.slider("Speechiness (Kerapatan Kata)", 0.0, 1.0, 0.2)
        danceability = st.slider("Danceability", 0.0, 1.0, 0.6)
        energy = st.slider("Energy", 0.0, 1.0, 0.7)
        loudness = st.slider("Loudness (dB)", -40.0, 5.0, -5.0)
    with col2:
        acousticness = st.slider("Acousticness", 0.0, 1.0, 0.1)
        instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0)
        liveness = st.slider("Liveness", 0.0, 1.0, 0.2)
        valence = st.slider("Valence", 0.0, 1.0, 0.5)
        tempo = st.slider("Tempo (BPM)", 50.0, 200.0, 100.0)

    if st.button("Analisis Keamanan Lagu", type="primary"):
        if logreg and scaler:
            import statsmodels.api as sm

            input_data = pd.DataFrame([[danceability, energy, loudness, speechiness,
                                        acousticness, instrumentalness, liveness, valence, tempo]],
                                      columns=['danceability', 'energy', 'loudness', 'speechiness',
                                               'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'])

            input_scaled = scaler.transform(input_data)
            input_reg = np.insert(input_scaled, 0, 1.0, axis=1)
            prob_explicit = logreg.predict(input_reg)[0]

            st.markdown("---")
            if prob_explicit >= 0.5:
                st.error("### HASIL: POTENSI KONTEN EKSPLISIT TINGGI")
                st.write(f"Model Regresi Logistik kami menghitung probabilitas sebesar **{prob_explicit*100:.1f}%** bahwa lagu ini mengandung bahasa kasar atau lirik dewasa.")
                st.caption("Catatan Analisis: Persentase tinggi umumnya didorong oleh kombinasi nilai Speechiness yang tinggi dan tingkat keakustikan yang rendah.")
            else:
                st.success("### HASIL: AMAN (NON-EKSPLISIT)")
                st.write(f"Model memprediksi lagu ini ramah keluarga dengan tingkat probabilitas eksplisit sebesar **{prob_explicit*100:.1f}%**.")
