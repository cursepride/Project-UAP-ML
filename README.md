# Project-UAP-ML
Proyek ini adalah sistem klasifikasi penyakit mata (ocular disease classification) menggunakan teknik deep learning berbasis Convolutional Neural Network (CNN). Tujuannya adalah mendeteksi delapan kategori penyakit mata dari gambar fundus (bagian belakang mata) secara otomatis, yang dapat membantu diagnosis dini oleh dokter mata. Proyek ini memanfaatkan dataset ODIR-5K dari Kaggle dan membandingkan performa tiga arsitektur model: CNN sederhana (dari setup dasar), MobileNetV2 (transfer learning ringan), dan EfficientNetB0 (transfer learning efisien).
Proyek mencakup tahap-tahap lengkap: pengunduhan dataset via Kaggle API, preprocessing data (termasuk balancing dan augmentasi), training model dengan fase pemanasan (warm-up) dan fine-tuning, serta evaluasi menggunakan metrik seperti accuracy, precision, recall, F1-score, dan confusion matrix. Kode dirancang untuk reproducibility dengan penguncian seed random (np.random.seed(42), dll.). Hasil model disimpan sebagai file .h5 (untuk model) dan .pkl (untuk label encoder), yang siap di-deploy ke aplikasi seperti VS Code atau mobile app.
Penjelasan Dataset dan Preprocessing
Dataset

Sumber: Dataset Ocular Disease Intelligent Recognition (ODIR-5K) dari Kaggle. Dataset ini berisi sekitar 5.000 gambar mata dari pasien di China, dikumpulkan untuk mendeteksi penyakit mata umum.
Komposisi:
Gambar: Resolusi bervariasi, tapi di-resize menjadi 224x224 pixel untuk input model.
Label: 8 kelas utama berdasarkan diagnosis:
N: Normal (mata sehat)
D: Diabetes (retinopati diabetik)
G: Glaucoma
C: Cataract (katarak)
A: Age-related Macular Degeneration (AMD)
H: Hypertension (retinopati hipertensi)
M: Myopia (miopia patologis)
O: Other (penyakit mata lain-lain)

Data tambahan: File CSV (full_df.csv atau serupa) berisi metadata seperti usia pasien, jenis kelamin, dan label one-hot encoding untuk setiap mata (kiri/kanan).

Ukuran: Awalnya tidak seimbang (misalnya, kelas 'N' dan 'D' lebih banyak daripada yang lain). Total sampel setelah filtering: Sekitar 3.000-4.000 gambar unik, tapi ditingkatkan menjadi 12.000 sampel setelah oversampling.
Lisensi: Other (bukan open-source penuh, tapi tersedia untuk penelitian).

Preprocessing
Preprocessing dilakukan secara konsisten di ketiga file untuk memastikan data siap diproses oleh model. Langkah-langkah utama:

Load dan Filtering Data:
Baca CSV menggunakan pd.read_csv().
Mapping label utama menggunakan fungsi get_label(row) untuk memilih label dominan dari kolom one-hot ('N', 'D', dll.).
Filter hanya gambar yang benar-benar ada di folder (preprocessed_images atau ODIR-5K_Training_Dataset) menggunakan os.path.exists() untuk menghindari error file not found.

Balancing Data (Oversampling):
Dataset awal tidak seimbang, sehingga digunakan sklearn.utils.resample() untuk oversampling setiap kelas menjadi 1.500 sampel (total 12.000 sampel).
Hasil: Distribusi seimbang, dicek dengan df_balanced['target'].value_counts().

Split Data:
Gunakan train_test_split() dari scikit-learn: 85% train, 15% validation, dengan stratify berdasarkan label untuk menjaga distribusi kelas.

Augmentasi dan Preprocessing Gambar:
ImageDataGenerator dari Keras: Rotasi 20°, horizontal flip, zoom 10% untuk train; hanya normalisasi untuk validation.
Penajaman Medis: Fungsi custom apply_clahe_medis(img) menggunakan CLAHE (Contrast Limited Adaptive Histogram Equalization) dari OpenCV untuk meningkatkan kontras pada gambar medis (konversi RGB ke LAB, apply CLAHE pada channel L, lalu kembali ke RGB).
Normalisasi: preprocess_input() dari model base (misalnya, MobileNetV2 atau EfficientNet) untuk scaling pixel ke [-1, 1] atau serupa.
Batch size: 32, target size: 224x224.

Label Encoding:
Gunakan LabelEncoder dari scikit-learn untuk mengubah label string ('N', 'D', dll.) menjadi indeks numerik.
Simpan encoder ke .pkl untuk deployment.


Preprocessing ini memastikan data robust terhadap variasi pencahayaan dan posisi mata, yang umum pada gambar medis.

#Penjelasan Ketiga Model yang Digunakan
Ketiga model menggunakan pendekatan transfer learning (kecuali CNN sederhana yang lebih basic) dengan training dua fase: Pemanasan (freeze base model) dan Fine-Tuning (unfreeze layer atas). Optimizer: Adam, Loss: Categorical Crossentropy, Metrics: Accuracy. Callback: EarlyStopping dan ReduceLROnPlateau untuk efisiensi.

CNN Sederhana (dari Model CNN.py):
Arsitektur: Ini adalah setup dasar CNN tanpa transfer learning spesifik (kode lebih fokus pada setup Kaggle dan download dataset). Model dibangun secara manual menggunakan tf.keras.models.Sequential() dengan layer seperti Conv2D, MaxPooling2D, Flatten, Dense, dan Dropout. Input shape: (224, 224, 3). Output: Dense(8, softmax) untuk 8 kelas.
Training: Epochs sekitar 10-20, batch 32. Tidak ada fase pemanasan spesifik, tapi mirip dengan model lain.
Kelebihan: Sederhana, mudah dimodifikasi, cepat train pada dataset kecil.
Kekurangan: Kurang efisien untuk dataset kompleks seperti gambar medis, rentan overfitting tanpa augmentasi kuat.

MobileNetV2 (dari untitled30.py):
Arsitektur: Transfer learning dari tf.keras.applications.MobileNetV2 (pre-trained on ImageNet, input 224x224). Tambahan layer: GlobalAveragePooling2D, Dropout(0.3), Dense(8, softmax).
Training:
Phase 1 (Pemanasan): Freeze base model, train 10 epochs dengan LR=1e-3.
Phase 2 (Fine-Tuning): Unfreeze layer atas (dari layer 100 ke atas), train 20 epochs dengan LR=1e-5.

Kelebihan: Ringan (ukuran model kecil, ~3-4 MB), cepat inferensi, cocok untuk deployment mobile.
Kekurangan: Kurang dalam pada fitur kompleks dibanding model lebih besar.

EfficientNetB0 (dari Salinan_dari_Untitled0.py):
Arsitektur: Transfer learning dari tf.keras.applications.EfficientNetB0 (pre-trained on ImageNet). Tambahan layer mirip MobileNet: GlobalAveragePooling2D, Dropout, Dense(8, softmax).
Training:
Phase 1 (Pemanasan): Freeze base, train epochs awal.
Phase 2 (Fine-Tuning): Unfreeze layer atas, train dengan LR rendah dan callback.

Kelebihan: Efisien (balance antara akurasi dan ukuran), menggunakan compound scaling untuk performa tinggi dengan parameter sedikit.
Kekurangan: Sedikit lebih lambat train daripada MobileNet, tapi lebih akurat.


Semua model disimpan sebagai model_name_ocular_final.h5.
Hasil Evaluasi dan Analisis Perbandingan
Evaluasi dilakukan pada validation set menggunakan model.predict(), diikuti classification_report dan confusion_matrix dari scikit-learn. Plot: Accuracy history, confusion matrix (raw dan normalized).
