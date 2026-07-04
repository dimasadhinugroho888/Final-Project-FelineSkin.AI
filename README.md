# FelineSkin.AI

FelineSkin.AI adalah aplikasi Streamlit untuk analisis awal kondisi kulit kucing menggunakan model ResNet18, Grad-CAM, penjelasan AI, chatbot khusus topik kucing, dan pencarian dokter/klinik hewan.

## Menjalankan Lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

Untuk fitur penjelasan AI dan chatbot, set secret berikut:

```bash
OPENROUTER_API_KEY=isi_api_key_openrouter
```

Atau buat file lokal `.streamlit/secrets.toml` berdasarkan `.streamlit/secrets.toml.example`.

## Deploy ke Streamlit Cloud

1. Push repo ke GitHub: `dimasadhinugroho888/Final-Project-FelineSkin.AI`.
2. Buat app baru di Streamlit Cloud dari repo tersebut.
3. Tambahkan secret di menu app settings:

```toml
OPENROUTER_API_KEY = "isi_api_key_openrouter"
```

Jangan commit file `.streamlit/secrets.toml` ke GitHub. File tersebut sudah masuk `.gitignore`.

## Catatan Medis

Aplikasi ini hanya untuk edukasi dan screening awal. Hasil model serta jawaban AI bukan diagnosis medis. Untuk kondisi berat, memburuk, luka terbuka, bernanah, menular, atau kucing terlihat lemas, segera konsultasikan ke dokter hewan.
