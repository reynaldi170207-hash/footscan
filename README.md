# FootScan 🦶
**Pengukur Ukuran Kaki Digital** — Computer Vision + Integral Numerik

---

## Deploy ke Render (Gratis)

### Langkah 1 — Upload ke GitHub
1. Buat repository baru di [github.com](https://github.com) (boleh private)
2. Upload semua file ini ke repository tersebut:
   ```
   app.py
   index.html
   requirements.txt
   Procfile
   render.yaml
   .gitignore
   ```

### Langkah 2 — Hubungkan ke Render
1. Buka [render.com](https://render.com) → Sign up / Login (gratis)
2. Klik **"New +"** → pilih **"Web Service"**
3. Pilih **"Connect a repository"** → hubungkan GitHub
4. Pilih repository FootScan kamu
5. Isi pengaturan:
   - **Name**: `footscan` (bebas)
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120 --preload`
6. Klik **"Create Web Service"**

### Langkah 3 — Tunggu Deploy
- Render akan otomatis install dependencies dan menjalankan server
- Setelah selesai (±3–5 menit), kamu mendapat URL seperti:
  `https://footscan.onrender.com`
- Buka URL tersebut → aplikasi langsung berjalan! ✅

---

## Jalankan Lokal

```bash
# Install dependencies
pip install -r requirements.txt

# Jalankan server
python app.py

# Buka browser
# http://localhost:5000
```

---

## Struktur File

```
footscan/
├── app.py           ← Backend Flask + Computer Vision
├── index.html       ← Frontend (semua halaman)
├── requirements.txt ← Python dependencies
├── Procfile         ← Perintah start untuk Render
├── render.yaml      ← Konfigurasi Render (opsional)
└── .gitignore
```

---

## Catatan Penting

- **Free tier Render**: server akan "tidur" setelah 15 menit tidak ada request.
  Request pertama setelah tidur butuh ~30 detik untuk bangun kembali (normal).
- **Upload gambar**: maksimal 10MB (JPG/PNG)
- **Analisis**: membutuhkan foto kaki dengan background kontras
