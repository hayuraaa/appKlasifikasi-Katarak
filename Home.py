import streamlit as st

# New Line
def new_line(n=1):
    for i in range(n):
        st.write("\n")

def main():
    # Judul
    st.markdown("<h1 align='center'> <b> Sistem Klasifikasi Penyakit Mata Katarak Menggunakan Random Forest dan Decision Tree</b> </h1>", unsafe_allow_html=True)
    new_line(1)
    st.markdown("Selamat datang! Aplikasi ini merupakan sistem prediksi penyakit mata katarak yang menggunakan algoritma Random Forest dengan Decision Tree. Sistem ini dirancang untuk membantu tenaga medis dalam melakukan diagnosis awal penyakit katarak berdasarkan gejala-gejala yang dialami pasien.", unsafe_allow_html=True)
    
    st.divider()
    
    #overview
    new_line()
    st.markdown("<h2 style='text-align: center; '>🏥 Gambaran Umum</h2>", unsafe_allow_html=True)
    new_line()
    
    st.markdown("""
    Dalam proses klasifikasi penyakit mata katarak, terdapat beberapa tahapan penting yang dilakukan:
    
    - **📊 Pengumpulan Data**: Mengumpulkan data pasien berdasarkan berbagai gejala dan faktor risiko katarak seperti usia, kekeruhan lensa, penurunan ketajaman penglihatan, dll.<br> <br>
    - **🔍 Preprocessing Data**: Proses standardisasi dan normalisasi data, termasuk encoding variabel kategorikal seperti jenis kelamin dan riwayat keluarga.<br> <br>
    - **🌳 Pemodelan Random Forest**: Menggunakan algoritma Random Forest yang terdiri dari multiple Decision Tree untuk melakukan klasifikasi.<br> <br>
    - **📈 Evaluasi Model**: Mengukur performa model menggunakan berbagai metrik seperti accuracy, precision, recall, dan F1-score.<br> <br>
    - **🔮 Prediksi**: Menggunakan model untuk memprediksi kemungkinan katarak pada pasien baru.<br> <br>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    Parameter penting dalam model Random Forest yang digunakan:
    
    - **🌲 Jumlah Pohon (n_estimators)**: Menentukan berapa banyak Decision Tree yang akan dibuat dalam Random Forest.<br> <br>
    - **📏 Kedalaman Maksimum (max_depth)**: Mengatur seberapa dalam setiap Decision Tree dapat tumbuh.<br> <br>
    - **📊 Minimum Sampel Split**: Jumlah minimum sampel yang diperlukan untuk membagi node internal.<br> <br>
    - **🎯 Kriteria Split**: Menggunakan entropy sebagai kriteria pemilihan fitur dalam pembagian node.<br> <br>
    - **⚖️ Class Weight**: Penanganan ketidakseimbangan kelas dalam data.<br> <br>
    """, unsafe_allow_html=True)
    new_line()
    
if __name__ == "__main__":
    main()