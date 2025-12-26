# --- KAMUS PENJELASAN PENYAKIT (Letakkan di luar button atau di bagian atas) ---
disease_info = {
    "A": {"nama": "AMD (Age-related Macular Degeneration)", "detail": "Penyakit akibat penuaan yang merusak makula (pusat penglihatan), menyebabkan pandangan tengah buram."},
    "C": {"nama": "Cataract (Katarak)", "detail": "Lensa mata menjadi keruh seperti berawan, menyebabkan penglihatan kabur dan sensitif terhadap cahaya."},
    "D": {"nama": "Diabetes Retinopathy", "detail": "Komplikasi diabetes yang merusak pembuluh darah di retina, bisa menyebabkan bercak hitam atau kebutaan."},
    "G": {"nama": "Glaucoma (Glaukoma)", "detail": "Kerusakan saraf optik akibat tekanan tinggi pada bola mata, seringkali merusak penglihatan tepi (tunnel vision)."},
    "H": {"nama": "Hypertension (Hipertensi)", "detail": "Kerusakan pembuluh darah retina akibat tekanan darah tinggi kronis (Hypertensive Retinopathy)."},
    "M": {"nama": "Myopia (Miopia Patologis)", "detail": "Rabun jauh tingkat ekstrem yang menyebabkan penipisan atau robekan pada jaringan retina."},
    "N": {"nama": "Normal", "detail": "Kondisi retina mata terlihat sehat, pembuluh darah normal, dan tidak ditemukan tanda-tanda kelainan."},
    "O": {"nama": "Others (Penyakit Lainnya)", "detail": "Ditemukan kelainan pada retina, namun tidak termasuk dalam 7 kategori utama di atas."}
}

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pickle
import os
from PIL import Image

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Ocular AI Diagnosis", layout="wide", page_icon="👁️")

# --- FUNGSI PREPROCESSING ---
def apply_clahe_medis(img_array, target_size):
    # Resize sesuai kebutuhan model masing-masing
    img_resized = cv2.resize(img_array, (target_size, target_size))
    
    # Konversi ke uint8 jika diperlukan
    if img_resized.max() <= 1.0:
        img_resized = (img_resized * 255).astype(np.uint8)
    
    # Proses CLAHE
    lab = cv2.cvtColor(img_resized, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    img_clahe = cv2.merge([l, a, b])
    img_final = cv2.cvtColor(img_clahe, cv2.COLOR_LAB2RGB)
    return img_final

# --- FUNGSI RECONSTRUCT MODEL (Solusi Error 2 Tensors) ---
def reconstruct_model(model_type):
    if model_type == "MobileNetV2":
        base_mn = tf.keras.applications.MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = base_mn(inputs, training=False) 
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        outputs = tf.keras.layers.Dense(8, activation='softmax')(x)
        return tf.keras.Model(inputs, outputs)
    
    elif model_type == "EfficientNet":
        base_eff = tf.keras.applications.EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = base_eff(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(8, activation='softmax')(x)
        return tf.keras.Model(inputs, outputs)
    return None

# --- UPDATE FUNGSI LOAD RESOURCES (VERSI CLEAN) ---
@st.cache_resource
def load_all_resources(model_display_name):
    # 1. Tentukan Path dan Target Size
    if model_display_name == "Base_CNN":
        target_size = 150 # Sesuaikan jika CNN Anda pakai 224
        model_path = "models/Base_CNN.h5"
    elif model_display_name == "MobileNetV2":
        target_size = 224
        model_path = "models/mobilenetv2_ocular_final.h5"
    else: # EfficientNet
        target_size = 224
        model_path = "models/odir_efficientnet_final.h5"

    # 2. Muat Model Secara Utuh
    # compile=False akan mengabaikan error optimizer yang sering muncul di laptop
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        st.error(f"Gagal memuat model {model_display_name}: {e}")
        return None, None, None

    # 3. Muat Label Encoder
    try:
        # Coba muat file pkl yang ada
        pkl_files = ['label_encoder_mn.pkl', 'label_encoder_efficientnet.pkl']
        le = None
        for p in pkl_files:
            if os.path.exists(p):
                with open(p, 'rb') as f:
                    le = pickle.read(f) if hasattr(pickle, 'read') else pickle.load(f)
                break
        
        if le is None:
            st.error("File Label Encoder (.pkl) tidak ditemukan!")
            
    except Exception as e:
        st.error(f"Error pada Label Encoder: {e}")
        return model, None, target_size

    return model, le, target_size

# --- SIDEBAR ---
st.sidebar.title("👁️ Ocular AI Setup")
selected_model_name = st.sidebar.selectbox("Pilih Arsitektur Model:", 
                                     ["Base_CNN", "MobileNetV2", "EfficientNet"])

# --- MAIN PANEL ---
st.title("Sistem AI Deteksi Penyakit Mata ODIR-5K")
st.write("Silakan unggah foto fundus mata untuk analisis.")

uploaded_file = st.file_uploader("Pilih gambar fundus...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load model & dapatkan target_size yang benar
    model, le, target_size = load_all_resources(selected_model_name)
    
    if model is not None:
        image = Image.open(uploaded_file).convert('RGB')
        img_array = np.array(image)
        
        # Lakukan preprocessing dengan target_size yang dinamis
        img_clahe = apply_clahe_medis(img_array, target_size)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Gambar Asli")
            st.image(image, use_container_width=True)
        with col2:
            st.subheader(f"Preprocessing CLAHE ({target_size}x{target_size})")
            st.image(img_clahe, use_container_width=True)

    # --- BAGIAN TOMBOL PREDIKSI ---
    if st.button("MULAI DIAGNOSA AI"):
        with st.spinner(f'Menganalisis dengan {selected_model_name}...'):
            # 1. Preprocessing Input
            # Pastikan img_clahe sudah dibuat di atas sebelum blok if ini
            img_input = img_clahe.astype('float32') / 255.0
            img_input = np.expand_dims(img_input, axis=0)
            
            # 2. Prediksi
            try:
                preds = model(img_input, training=False).numpy()
                    
                idx = np.argmax(preds)
                label_hasil = le.classes_[idx]
                confidence = np.max(preds) * 100
                
                # Ambil data dari kamus
                info = disease_info.get(label_hasil, {"nama": "Tidak Diketahui", "detail": "-"})

                st.divider()
                
                # Layout Hasil: Kiri untuk Teks, Kanan untuk Grafik
                res_col1, res_col2 = st.columns([1.2, 1])
                
                with res_col1:
                    # Tampilan Box Hasil yang Menarik
                    st.markdown(f"""
                    <div style="background-color: #161b22; padding: 25px; border-radius: 15px; border-left: 8px solid #28a745; border-right: 1px solid #30363d; border-top: 1px solid #30363d; border-bottom: 1px solid #30363d;">
                        <h5 style="color: #8b949e; margin-bottom: 10px; text-transform: uppercase;">Hasil Analisis AI:</h5>
                        <h2 style="color: #ffffff; margin-bottom: 5px;">{label_hasil} - {info['nama']}</h2>
                        <h3 style="color: #2ea043; margin-bottom: 20px;">Tingkat Keyakinan: {confidence:.2f}%</h3>
                        <hr style="border: 0.5px solid #30363d;">
                        <p style="color: #c9d1d9; font-size: 16px;"><b>Tentang Penyakit:</b><br>{info['detail']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with res_col2:
                    st.subheader("Probabilitas Tiap Kelas")
                    # Konversi hasil prediksi ke dictionary untuk grafik bar
                    chart_data = dict(zip(le.classes_, [float(p) for p in preds[0]]))
                    st.bar_chart(chart_data)
                    
            except Exception as e:
                st.error(f"Terjadi kesalahan saat prediksi: {e}")