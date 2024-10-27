import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go

def load_model():
    try:
        return joblib.load('FinalModel.joblib')
    except:
        return None

def show_feature_info():
    st.info("""
    ℹ️ **Panduan Pengisian Data:**
    
    1. **Usia**: Usia pasien (40-90 tahun)
    2. **Jenis Kelamin**: Pilih jenis kelamin pasien
    3. **Riwayat Keluarga**: Apakah ada anggota keluarga yang menderita katarak
    4. **Kekeruhan Lensa**: Tingkat kekeruhan lensa (0-10)
       - 0-3: Normal
       - 4-7: Kekeruhan Sedang 
       - 8-10: Kekeruhan Tinggi
    5. **Penurunan Ketajaman**: Tingkat penurunan ketajaman penglihatan (0-10)
    6. **Sensitivitas Cahaya**: Tingkat sensitivitas terhadap cahaya (0-10)
    7. **Perubahan Warna**: Apakah pasien mengalami perubahan persepsi warna
    8. **Penglihatan Ganda**: Apakah pasien mengalami penglihatan ganda
    9. **Tekanan Intraokular**: Tekanan dalam bola mata (normal: 12-22 mmHg)
    10. **Hasil Slitlamp**: Hasil pemeriksaan dengan alat slitlamp
    11. **Visus**: Ketajaman penglihatan (0-1)
       - 1.0: Normal
       - 0.8-0.9: Ringan
       - 0.5-0.7: Sedang
       - < 0.5: Berat
    """)

def main():
    st.title('🏥 Sistem Klasifikasi Katarak')
    
    # Informasi Umum
    st.markdown("""
    <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
        <h4>ℹ️ Tentang Sistem</h4>
        <p>Sistem ini menggunakan algoritma Random Forest untuk mengklasifikasikan kemungkinan katarak berdasarkan 
        gejala dan hasil pemeriksaan pasien. Model telah dilatih menggunakan dataset yang mencakup berbagai kasus 
        katarak dan non-katarak.</p>
    </div>
    """, unsafe_allow_html=True)
    
    model = load_model()
    
    if model is None:
        st.error('⚠️ Model belum tersedia. Silakan latih model terlebih dahulu.')
        return

    # Sidebar untuk input data
    st.sidebar.header('📝 Input Data Pasien')
    usia = st.sidebar.number_input('Usia', min_value=0, max_value=120, value=65)
    jenis_kelamin = st.sidebar.selectbox('Jenis Kelamin', ['Pria', 'Wanita'])
    riwayat_keluarga = st.sidebar.selectbox('Riwayat Keluarga', ['Ya', 'Tidak'])
    kekeruhan_lensa = st.sidebar.slider('Kekeruhan Lensa', 0.0, 10.0, 5.0)
    penurunan_ketajaman = st.sidebar.slider('Penurunan Ketajaman', 0, 10, 5)
    sensitivitas_cahaya = st.sidebar.slider('Sensitivitas Cahaya', 0, 10, 5)
    perubahan_warna = st.sidebar.selectbox('Perubahan Warna', ['Ya', 'Tidak'])
    penglihatan_ganda = st.sidebar.selectbox('Penglihatan Ganda', ['Ya', 'Tidak'])
    tekanan_intraokular = st.sidebar.slider('Tekanan Intraokular', 10.0, 30.0, 18.0)
    hasil_slitlamp = st.sidebar.selectbox('Hasil Slitlamp', ['Normal', 'Abnormal'])
    visus = st.sidebar.slider('Visus', 0.0, 1.0, 0.5)

    # Tombol prediksi
    submitted = st.sidebar.button('🔍 Analisis Prediksi', use_container_width=True)

    # Tampilkan panduan pengisian di area utama
    show_feature_info()

    if submitted:
        # Prepare input data
        input_data = {
            'usia': usia,
            'jenis_kelamin': jenis_kelamin,
            'riwayat_keluarga': riwayat_keluarga,
            'kekeruhan_lensa': kekeruhan_lensa,
            'penurunan_ketajaman': penurunan_ketajaman,
            'sensitivitas_cahaya': sensitivitas_cahaya,
            'perubahan_warna': perubahan_warna,
            'penglihatan_ganda': penglihatan_ganda,
            'tekanan_intraokular': tekanan_intraokular,
            'hasil_slitlamp': hasil_slitlamp,
            'visus': visus
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Encode categorical variables
        le = LabelEncoder()
        categorical_columns = ['jenis_kelamin', 'riwayat_keluarga', 'perubahan_warna', 
                               'penglihatan_ganda', 'hasil_slitlamp']
        for col in categorical_columns:
            input_df[col] = le.fit_transform(input_df[col])

        # Make prediction with loading spinner
        with st.spinner('Menganalisis data...'):
            prediction = model.predict(input_df)
            probability = model.predict_proba(input_df)

        # Display results dengan styling
        st.subheader('📊 Hasil Analisis:')
        
        # Buat box hasil dengan warna sesuai prediksi
        if prediction[0] == 1:
            st.markdown(f"""
            <div style='background-color: #ff4b4b; padding: 20px; border-radius: 10px; color: white;'>
                <h3>⚠️ TERDETEKSI KATARAK</h3>
                <p>Probabilitas: {probability[0][1]:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background-color: #00cc44; padding: 20px; border-radius: 10px; color: white;'>
                <h3>✅ TIDAK TERDETEKSI KATARAK</h3>
                <p>Probabilitas: {probability[0][0]:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
            
        # Visualisasi Probabilitas
        st.subheader('📈 Visualisasi Probabilitas')
        prob_df = pd.DataFrame({
            'Kelas': ['Tidak Katarak', 'Katarak'],
            'Probabilitas': [probability[0][0], probability[0][1]]
        })
        
        fig_bar = px.bar(prob_df, x='Kelas', y='Probabilitas',
                         title='Perbandingan Probabilitas Diagnosis',
                         color='Kelas',
                         color_discrete_map={'Tidak Katarak': '#00cc44', 'Katarak': '#ff4b4b'},
                         text='Probabilitas')
        fig_bar.update_traces(texttemplate='%{text:.2%}', textposition='outside')
        fig_bar.update_layout(
            yaxis_tickformat='.0%',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis_gridcolor='rgba(0,0,0,0.1)'
        )
        st.plotly_chart(fig_bar)

        # Tambahkan rekomendasi
        st.subheader('💡 Rekomendasi:')
        if prediction[0] == 1:
            st.warning("""
            - Segera konsultasikan dengan dokter mata untuk pemeriksaan lebih lanjut
            - Hindari aktivitas yang membutuhkan ketajaman penglihatan tinggi
            - Gunakan pelindung mata saat beraktivitas di luar ruangan
            - Perhatikan gejala yang mungkin bertambah parah
            """)
        else:
            st.success("""
            - Lakukan pemeriksaan mata rutin setiap 6-12 bulan
            - Jaga pola hidup sehat dan konsumsi makanan bergizi
            - Gunakan pelindung mata saat beraktivitas di bawah sinar matahari
            - Segera konsultasi jika muncul gejala penglihatan abnormal
            """)

    # Catatan penting di bagian bawah
    st.markdown("""
    ---
    **Catatan Penting:**
    - Hasil prediksi ini hanya bersifat pendukung diagnosis dan tidak menggantikan pemeriksaan medis langsung
    - Konsultasikan selalu dengan dokter mata untuk diagnosis yang akurat
    - Sistem ini memiliki tingkat akurasi tertentu dan masih mungkin terjadi kesalahan prediksi
    """)

if __name__ == '__main__':
    main()