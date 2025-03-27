# Genosense - Palm Oil Monitoring System

Genosense adalah aplikasi web berbasis Flask untuk memantau infeksi Ganoderma boninense pada tanaman kelapa sawit menggunakan teknologi pembelajaran mesin dan cellular automata.

## Fitur Utama

- Peta interaktif untuk visualisasi data infeksi
- Analisis pola penyebaran infeksi
- Visualisasi tren perkembangan infeksi
- Prediksi penyebaran infeksi menggunakan model Cellular Automata
- Upload dan analisis citra multispektral secara otomatis

## Dependensi

- Python 3.11
- Flask 2.3.3
- SQLAlchemy 2.0.23
- PostgreSQL
- TensorFlow 2.15.0
- NumPy, scikit-learn, scikit-image
- Lihat file `dependencies.txt` untuk daftar lengkap

## Cara Menggunakan

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/username/genosense.git
cd genosense

# Buat virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependensi
pip install -r dependencies.txt
```

### 2. Konfigurasi Database

```bash
# Set environment variables
export DATABASE_URL=postgresql://username:password@localhost:5432/genosense
export SESSION_SECRET=your-secret-key

# Windows
set DATABASE_URL=postgresql://username:password@localhost:5432/genosense
set SESSION_SECRET=your-secret-key
```

### 3. Menjalankan Aplikasi

```bash
# Jalankan dengan Gunicorn (untuk produksi)
gunicorn --bind 0.0.0.0:5000 --reuse-port main:app

# Atau menggunakan Flask (untuk development)
python main.py
```

Aplikasi akan berjalan di http://localhost:5000

### 4. GitHub Actions

Repository ini sudah dilengkapi dengan GitHub Actions workflow untuk otomatis testing dan deployment. Lihat file `.github/workflows/python-app.yml` untuk detail konfigurasi.

## Struktur Direktori

```
genosense/
├── .github/workflows/       # GitHub Actions workflow
├── instance/                # Database instance files
├── ml_utils/                # Modul utilitas machine learning
├── model_utils/             # Utilitas dan helper untuk model database
├── static/                  # Asset statis (CSS, JS, gambar)
├── templates/               # Template HTML
├── uploads/                 # Direktori unggahan gambar
├── app.py                   # Konfigurasi aplikasi Flask
├── cellular_automata.py     # Implementasi model Cellular Automata
├── main.py                  # Entry point aplikasi
├── ml_models.py             # Implementasi model machine learning
├── models.py                # Model database
└── routes.py                # Endpoint dan route aplikasi
```

## Kontribusi

Kontribusi selalu diterima. Silakan fork repository ini, buat perubahan, dan kirim pull request.