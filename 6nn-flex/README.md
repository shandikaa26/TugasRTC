# Neural Network Flexible Dataset Analyzer

Aplikasi analisis dengan jaringan saraf (neural network) berbasis Rust yang dapat menganalisis berbagai dataset tabular. Dirancang untuk fleksibilitas maksimum dengan antarmuka pengguna yang intuitif.

## Fitur Utama

- **Analisis Dataset Fleksibel**: Mendukung berbagai format dataset CSV dengan jumlah kolom yang fleksibel
- **Konfigurasi Dataset**:
  - Pilihan kolom label dinamis
  - Deteksi otomatis delimiter (pemisah) file CSV
  - Dukungan untuk file dengan atau tanpa header
  - Opsi penanganan error dalam dataset
- **Model Jaringan Saraf yang Dapat Disesuaikan**:
  - Jumlah layer tersembunyi yang dapat dikonfigurasi
  - Jumlah neuron per layer yang dapat disesuaikan
  - Learning rate yang dapat diatur
  - Jumlah epoch yang dapat dikonfigurasi
- **Visualisasi Real-time**:
  - Grafik akurasi selama proses training
  - Grafik loss selama proses training
- **Antarmuka Pengguna yang Intuitif**:
  - Preview dataset
  - Pemilihan file dengan mudah
  - Status training yang informatif

## Persyaratan Teknis

- Rust 2021 edition atau yang lebih baru
- Cargo package manager

## Cara Penggunaan

### Menjalankan Aplikasi

```bash
cargo run --release
```

### Langkah-langkah Analisis Dataset

1. **Pilih Dataset**: Klik tombol "Browse for Dataset" dan pilih file CSV yang ingin dianalisis.
2. **Konfigurasi Dataset**:
   - Periksa apakah baris pertama berisi header
   - Pilih delimiter yang sesuai jika tidak terdeteksi secara otomatis
   - Pilih kolom yang berisi label (target) untuk prediksi
   - Terapkan konfigurasi dengan mengklik "Apply Configuration"
3. **Atur Parameter Jaringan Saraf**:
   - Tetapkan jumlah epoch
   - Atur jumlah layer tersembunyi
   - Atur jumlah neuron per layer
   - Tentukan learning rate
4. **Mulai Training**: Klik tombol "Start Training" untuk memulai proses training
5. **Pantau Kemajuan**: Lihat grafik akurasi dan loss yang diperbarui secara real-time

## Struktur Proyek

```
.
├── src/                # Kode sumber Rust
│   ├── main.rs         # Implementasi inti dan thread training
│   ├── frontend_qt.rs  # Implementasi antarmuka pengguna
│   └── data/           # Contoh dataset (jika disertakan)
```

## Tipe Dataset yang Didukung

Aplikasi ini mendukung dataset tabular dalam format CSV dengan karakteristik berikut:
- File dengan atau tanpa header
- Berbagai jenis delimiter (koma, titik koma, tab, pipe)
- Kolom berisi nilai numerik (nilai non-numerik akan ditangani sesuai konfigurasi)
- Semua kolom kecuali kolom label digunakan sebagai fitur
- Label diharapkan berupa nilai 0 atau 1 (klasifikasi biner)

## Tips Penggunaan

- Dataset harus berupa nilai numerik
- Gunakan fitur preview untuk memastikan dataset terbaca dengan benar
- Untuk performa terbaik pada dataset besar, pilih jumlah epoch yang sesuai

## Lisensi

MIT License 