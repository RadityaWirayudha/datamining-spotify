import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Konfigurasi Halaman
st.set_page_config(page_title="Spotify Audio Analytics", layout="wide")

# Load Data, Model, dan Scaler
@st.cache_data
def load_dataset():
    return pd.read_csv('spotify_deployed.csv')

try:
    df = load_dataset()
    with open('scaler_spotify.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('kmeans_spotify.pkl', 'rb') as f:
        kmeans = pickle.load(f)
except FileNotFoundError:
    st.error("File model atau dataset belum ditemukan. Pastikan proses export (pickle) sudah dijalankan.")

# Navigasi Sidebar
st.sidebar.title("Navigasi Aplikasi")
pilihan_halaman = st.sidebar.radio("Pilih Halaman Menu:",
                                 ["Beranda & Pemahaman Bisnis",
                                  "Generator Playlist (Berdasarkan Mood)",
                                  "Prediksi Segmen Lagu Baru"])

# --- HALAMAN 1: BERANDA ---
if pilihan_halaman == "Beranda & Pemahaman Bisnis":
    st.title("Sistem Analisis Karakteristik Audio Spotify")
    st.write("Aplikasi ini merupakan implementasi dari model Data Mining (K-Means Clustering) untuk mengelompokkan karakteristik audio secara objektif tanpa bias label genre.")

    st.subheader("Profil Segmen (Cluster)")
    st.write("- **Cluster 0:** Segmen Relaksasi dan Fokus (Didominasi lagu akustik, tempo lambat, energi rendah)")
    st.write("- **Cluster 1:** Segmen Agresif dan Intens (Didominasi lagu berenergi tinggi, keras, tempo cepat)")
    # Removed Cluster 2 description since it's now 2 clusters

    st.info("Gunakan navigasi di sebelah kiri untuk mencoba sistem rekomendasi dan prediksi.")

# --- HALAMAN 2: GENERATOR PLAYLIST ---
elif pilihan_halaman == "Generator Playlist (Berdasarkan Mood)":
    st.title("Smart Auto-Playlist")
    st.write("Sistem akan merekomendasikan lagu berdasarkan karakteristik audio yang sesuai dengan aktivitas Anda.")

    pilihan_mood = st.selectbox("Pilih Aktivitas atau Mood Anda saat ini:",
                                ["-- Pilih --",
                                 "Membaca Buku",
                                 "Pengantar Tidur",
                                 "Fokus Belajar",
                                 "Meditasi atau Yoga",
                                 "Olahraga", # Combined "Olahraga dan Gym" and "Lari Pagi"
                                 "Membangkitkan Adrenalin",
                                 "Pesta dan Kumpul Teman",
                                 "Roadtrip atau Perjalanan",
                                 "Menaikkan Mood (Good Vibes)"])

    if st.button("Buat Playlist"):
        if pilihan_mood == "-- Pilih --":
            st.warning("Silakan pilih aktivitas terlebih dahulu.")
        else:
            # Kelompok Santai (Cluster 0)
            if pilihan_mood in ["Membaca Buku", "Pengantar Tidur", "Fokus Belajar", "Meditasi atau Yoga"]:
                target = 0

            # Kelompok Agresif (Cluster 1)
            elif pilihan_mood in ["Olahraga", "Membangkitkan Adrenalin", "Pesta dan Kumpul Teman", "Roadtrip atau Perjalanan", "Menaikkan Mood (Good Vibes)"]:
                target = 1

            # Removed the else case for Cluster 2 as there are only 2 clusters now

            df_hasil = df[df['cluster'] == target]

            # Mengambil sampel 5 lagu
            playlist = df_hasil.sample(5)[['track_name', 'artists', 'popularity', 'track_genre']]
            playlist.columns = ['Judul Lagu', 'Artis', 'Skor Popularitas', 'Genre Asli']
            playlist.reset_index(drop=True, inplace=True)

            st.success(f"Berikut adalah rekomendasi lagu untuk '{pilihan_mood}':")
            st.table(playlist)
            st.caption(f"Catatan Developer: Data ditarik otomatis dari Cluster {target} murni berdasarkan metrik audio.")

# --- HALAMAN 3: PREDIKSI LAGU BARU (Implementasi Model) ---
elif pilihan_halaman == "Prediksi Segmen Lagu Baru":
    st.title("Prediksi Segmen Lagu (Clustering)")
    st.write("Masukkan parameter metrik audio dari sebuah lagu baru untuk melihat ke segmen mana lagu tersebut akan dikelompokkan oleh mesin.")

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

    if st.button("Prediksi Cluster"):
        # Menyusun data input sesuai urutan fitur saat training
        input_data = pd.DataFrame([[danceability, energy, loudness, speechiness,
                                    acousticness, instrumentalness, liveness, valence, tempo]],
                                  columns=['danceability', 'energy', 'loudness', 'speechiness',
                                           'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'])

        # Standarisasi input menggunakan scaler yang di-load
        input_scaled = scaler.transform(input_data)

        # Prediksi menggunakan model K-Means
        hasil_prediksi = kmeans.predict(input_scaled)[0]

        # Output hasil
        if hasil_prediksi == 0:
            st.success("Hasil Prediksi: Lagu ini masuk ke CLUSTER 0 (Segmen Relaksasi & Akustik)")
        elif hasil_prediksi == 1:
            st.error("Hasil Prediksi: Lagu ini masuk ke CLUSTER 1 (Segmen Agresif & Intens)")
        # Removed the else case for Cluster 2 as there are only 2 clusters now
