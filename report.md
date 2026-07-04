# Laporan Analisis Kode FelineSkin.AI

Tanggal audit: 2026-07-04

## Ringkasan Proyek

FelineSkin.AI adalah aplikasi Streamlit untuk mendeteksi kondisi kulit kucing dari gambar. Alur utama aplikasi:

1. Pengguna upload gambar.
2. Aplikasi mencoba memvalidasi apakah gambar adalah kucing atau close-up tekstur.
3. Model ResNet18 melakukan klasifikasi ke 4 kelas:
   - `Flea_Allergy`
   - `Health`
   - `Ringworm`
   - `Scabies`
4. Aplikasi menampilkan probabilitas, Grad-CAM, penjelasan AI dari OpenRouter, dan Google Maps untuk mencari dokter/klinik hewan.

File utama:

- `app.py`: seluruh logic aplikasi, inference model, UI Streamlit, OpenRouter, Grad-CAM, dan Google Maps.
- `cat_skin_disease_model.pth`: checkpoint model PyTorch.
- `class_names.txt`: daftar kelas model.
- `requirements.txt`: dependensi Python.
- `Dockerfile`: konfigurasi container untuk deploy.

Validasi cepat yang sudah dilakukan:

- `python -m py_compile app.py`: sukses, tidak ada syntax error.
- Checkpoint `cat_skin_disease_model.pth`: berbentuk `OrderedDict`, memiliki `fc.weight` ukuran `(4, 512)` dan `fc.bias` ukuran `(4,)`, cocok dengan 4 kelas di `class_names.txt`.

## Temuan Bug dan Risiko

### 1. Pesan error model menyesatkan

Lokasi: `app.py` baris 219-223

Kode saat ini menangkap semua error saat `model.load_state_dict(...)`, lalu selalu menampilkan:

`Model tidak ditemukan!`

Masalahnya, error tidak selalu karena file model hilang. Bisa juga karena:

- arsitektur model tidak cocok,
- jumlah kelas berubah,
- file checkpoint rusak,
- format checkpoint berbeda,
- versi PyTorch bermasalah.

Dampak: debugging menjadi sulit karena penyebab asli disembunyikan.

Rekomendasi:

- Tangkap `FileNotFoundError` secara khusus.
- Untuk error lain, tampilkan pesan teknis ringkas ke developer atau log.

Contoh arah perbaikan:

```python
try:
    state_dict = torch.load("cat_skin_disease_model.pth", map_location="cpu")
    model.load_state_dict(state_dict)
except FileNotFoundError:
    st.error("File model cat_skin_disease_model.pth tidak ditemukan.")
    return
except RuntimeError as e:
    st.error("Model ditemukan, tetapi strukturnya tidak cocok dengan aplikasi.")
    st.exception(e)
    return
```

### 2. Banyak `except:` kosong menyembunyikan bug

Lokasi:

- `app.py` baris 13-16
- `app.py` baris 67-79
- `app.py` baris 98-102
- `app.py` baris 219-223

Masalah:

`except:` tanpa jenis error akan menangkap semua hal, termasuk bug coding yang seharusnya terlihat. Pada fungsi OpenRouter, semua error API, JSON parse, koneksi, timeout, auth, dan struktur respons disamakan menjadi `AI gagal merespon`.

Dampak:

- Sulit membedakan API key kosong, model tidak tersedia, rate limit, timeout, atau format respons berubah.
- Bug tersembunyi dan aplikasi terlihat "gagal biasa".

Rekomendasi:

- Gunakan exception spesifik.
- Simpan detail error di log.
- Tampilkan pesan user-friendly untuk pengguna.

### 3. Tidak ada validasi `OPENROUTER_API_KEY`

Lokasi: `app.py` baris 13-16 dan 31-33

Jika API key tidak tersedia, header tetap dikirim sebagai:

`Authorization: Bearer None`

Dampak:

- Request pasti gagal.
- Pengguna hanya melihat `AI gagal merespon`, tanpa tahu API key belum disetel.

Rekomendasi:

- Cek API key sebelum request.
- Jika kosong, tampilkan pesan konfigurasi.

Contoh:

```python
if not OPENROUTER_API_KEY:
    return "API key OpenRouter belum dikonfigurasi."
```

### 4. Detektor kucing memakai `torch.hub` dan bisa gagal saat offline/deploy

Lokasi:

- `app.py` baris 86-93

Model ResNet18 dimuat dari:

```python
torch.hub.load("pytorch/vision:v0.10.0", "resnet18", ...)
```

Masalah:

- Saat pertama kali jalan, `torch.hub` bisa butuh internet untuk download repo/model.
- Di Docker atau hosting tanpa akses internet stabil, aplikasi bisa gagal.
- Versi `pytorch/vision:v0.10.0` sudah lama.

Dampak:

- Deploy bisa lambat atau gagal.
- Startup aplikasi tergantung jaringan.

Rekomendasi:

- Gunakan `torchvision.models.resnet18` langsung dari package `torchvision`.
- Untuk detector ImageNet, gunakan weights resmi dari torchvision dan pastikan cache/dependensi tersedia.
- Pertimbangkan menghapus detektor ImageNet jika validasi close-up lebih penting daripada foto seluruh kucing.

### 5. Deteksi kucing hanya memakai 5 indeks ImageNet

Lokasi: `app.py` baris 133-148

Kode hanya memakai indeks:

```python
cat_indices = [281, 282, 283, 284, 285]
```

Masalah:

ImageNet memiliki lebih banyak kelas kucing domestik, misalnya tiger cat, Persian cat, Siamese cat, Egyptian cat, tabby, dan lainnya. Dengan hanya 5 indeks, banyak foto kucing valid bisa ditolak.

Dampak:

- False negative tinggi.
- Pengguna upload foto kucing tetapi aplikasi berkata bukan kucing.

Rekomendasi:

- Tambahkan semua indeks ImageNet yang relevan untuk kucing.
- Atau gunakan model/object detector yang memang mendeteksi `cat`, misalnya YOLO/COCO.
- Untuk aplikasi penyakit kulit, pertimbangkan mode "foto close-up kulit" sebagai jalur utama, karena foto penyakit sering tidak memperlihatkan seluruh tubuh kucing.

### 6. Deteksi close-up berbasis threshold manual mudah salah

Lokasi: `app.py` baris 153-171

Fungsi `is_closeup_texture` memakai Laplacian variance dan Canny edge density.

Masalah:

- Tekstur kain, karpet, rumput, kulit manusia, atau gambar tajam lain bisa lolos sebagai close-up.
- Foto kulit kucing yang blur atau terlalu terang bisa ditolak.

Dampak:

- Input non-kucing bisa tetap diproses.
- Hasil prediksi bisa tampak meyakinkan padahal input tidak valid.

Rekomendasi:

- Tambahkan validasi berbasis model untuk "cat skin/fur close-up" vs "non-cat".
- Minimal tambahkan peringatan kuat jika mode close-up dipakai.
- Simpan metrik `lap_var` dan `edge_density` untuk evaluasi threshold.

### 7. Grad-CAM menggunakan `register_backward_hook` yang sudah deprecated

Lokasi: `app.py` baris 182-183

`register_backward_hook` sudah deprecated di PyTorch modern dan bisa memberi perilaku tidak lengkap pada graph tertentu.

Dampak:

- Warning runtime.
- Grad-CAM bisa tidak akurat atau rusak pada versi PyTorch tertentu.

Rekomendasi:

Gunakan:

```python
register_full_backward_hook
```

### 8. Potensi warna overlay Grad-CAM terbalik

Lokasi: `app.py` baris 274-278

`cv2.applyColorMap` menghasilkan gambar BGR, sedangkan `img_np` dari PIL adalah RGB. Saat `cv2.addWeighted(img_np, 0.6, heat, 0.4, 0)` dilakukan, channel warna dicampur antara RGB dan BGR. Setelah itu baru dikonversi `BGR2RGB`.

Dampak:

- Warna heatmap/overlay bisa tidak sesuai.
- Area "merah" yang dijelaskan ke pengguna bisa terlihat berbeda.

Rekomendasi:

- Konversi heatmap BGR ke RGB sebelum overlay.

Contoh:

```python
heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
overlay = cv2.addWeighted(img_np, 0.6, heat, 0.4, 0)
```

### 9. Mutasi tensor activation pada Grad-CAM bisa berisiko

Lokasi: `app.py` baris 194-199

Kode:

```python
act = acts[0][0]
for i in range(act.shape[0]):
    act[i] *= pooled[i]
```

Masalah:

`act` adalah tensor dari activation hook. Operasi in-place bisa membuat debugging autograd lebih sulit dan berpotensi memengaruhi graph.

Rekomendasi:

Gunakan operasi non-in-place:

```python
weighted = act * pooled[:, None, None]
heat = torch.mean(weighted, dim=0)
```

### 10. Tidak ada `torch.no_grad()` untuk detektor Grad-CAM sudah benar, tapi inference utama memuat model ulang setiap rerun

Lokasi: `app.py` baris 215-225

Streamlit melakukan rerun saat input berubah. `load_model()` memang di-cache, tetapi `model.fc = ...` dan `model.load_state_dict(...)` dijalankan di dalam `main()`, sehingga bisa terjadi berulang.

Dampak:

- Aplikasi lebih lambat.
- State model yang di-cache dimodifikasi ulang setiap rerun.

Rekomendasi:

- Pindahkan seluruh proses load model utama, termasuk penggantian `fc` dan `load_state_dict`, ke fungsi `@st.cache_resource`.

### 11. Model AI OpenRouter memakai daftar model yang mungkin tidak stabil

Lokasi: `app.py` baris 58-64

Daftar model free dapat berubah, dibatasi, atau tidak tersedia sewaktu-waktu. Beberapa nama model juga terlihat berisiko berubah/eksperimental.

Dampak:

- Fitur penjelasan AI tiba-tiba gagal tanpa perubahan kode.

Rekomendasi:

- Simpan model utama di konfigurasi.
- Tambahkan fallback yang jelas.
- Tampilkan error HTTP dari OpenRouter untuk admin/developer.
- Tambahkan `res.raise_for_status()` dan cek respons JSON.

### 12. Prompt AI belum membatasi klaim medis

Lokasi: `app.py` baris 36-56

Prompt meminta penjelasan penyakit dan penanganan awal, tetapi belum secara eksplisit melarang diagnosis pasti atau rekomendasi obat spesifik.

Dampak:

- AI bisa memberi saran medis terlalu percaya diri.
- Risiko keamanan untuk hewan.

Rekomendasi:

Tambahkan instruksi:

- jangan menyatakan diagnosis pasti,
- jangan memberi dosis obat,
- sarankan dokter hewan untuk kasus parah/menular/memburuk,
- jawab dalam bahasa Indonesia,
- gunakan format ringkas dan aman.

### 13. Tidak ada validasi ukuran/tipe file upload secara ketat

Lokasi: `app.py` baris 227-230

Streamlit membatasi ekstensi, tetapi belum ada validasi:

- ukuran file,
- gambar corrupt,
- resolusi sangat besar,
- EXIF orientation.

Dampak:

- Gambar besar bisa memperlambat aplikasi.
- File corrupt bisa membuat error tidak tertangani.

Rekomendasi:

- Bungkus `Image.open` dengan try-except.
- Batasi ukuran file.
- Gunakan `ImageOps.exif_transpose(img)`.

### 14. Google Maps embed membuat query dari input mentah

Lokasi: `app.py` baris 303-313

Query dibuat dengan `replace(" ", "+")`. Ini belum melakukan URL encoding penuh.

Dampak:

- Karakter khusus seperti `&`, `?`, `#`, koma, atau slash bisa membuat URL rusak.

Rekomendasi:

Gunakan:

```python
from urllib.parse import quote_plus
encoded_query = quote_plus(query)
```

### 15. Dependensi tidak dipin versinya

Lokasi: `requirements.txt`

Semua dependency tidak memiliki versi:

```txt
streamlit
torch
torchvision
pillow
numpy
opencv-python-headless
requests
```

Dampak:

- Build hari ini dan build minggu depan bisa menghasilkan environment berbeda.
- Risiko aplikasi rusak karena update besar PyTorch/Streamlit/OpenCV.

Rekomendasi:

- Pin versi dependency yang sudah terbukti jalan.
- Minimal gunakan range versi aman.

### 16. Docker image kemungkinan besar menjadi sangat besar

Lokasi: `Dockerfile`

`torch` dan `torchvision` dari pip bisa menarik package besar. File model juga sekitar 44 MB. Docker image kemungkinan besar membengkak.

Dampak:

- Deploy lebih lambat.
- Cold start lebih berat.

Rekomendasi:

- Gunakan base image dan dependency CPU-only PyTorch yang sesuai.
- Tambahkan `.dockerignore`.
- Jangan ikutkan file yang tidak diperlukan.

### 17. Tidak ada test otomatis

Saat ini tidak ditemukan file test.

Dampak:

- Perubahan kecil pada preprocessing, class names, model loading, atau Grad-CAM sulit divalidasi.
- Bug runtime baru ketahuan setelah aplikasi dijalankan manual.

Rekomendasi test minimal:

- Test `load_class_names()`.
- Test `preprocess()` menghasilkan shape `[1, 3, 224, 224]`.
- Test checkpoint cocok dengan jumlah kelas.
- Test URL Google Maps memakai encoding benar.
- Test `get_ai_explanation()` saat API key kosong.

## Prioritas Perbaikan

### Prioritas Tinggi

1. Validasi `OPENROUTER_API_KEY` sebelum request.
2. Perbaiki error handling model loading agar tidak semua error menjadi `Model tidak ditemukan`.
3. Pindahkan load model utama sepenuhnya ke `@st.cache_resource`.
4. Ganti `register_backward_hook` ke `register_full_backward_hook`.
5. Perbaiki warna overlay Grad-CAM.
6. Pin versi dependency.

### Prioritas Menengah

1. Perbaiki deteksi kucing agar tidak hanya memakai 5 indeks ImageNet.
2. Tambahkan validasi upload gambar.
3. Gunakan URL encoding untuk Google Maps.
4. Tambahkan log/debug mode untuk error OpenRouter.
5. Buat unit test minimal.

### Prioritas Rendah

1. Rapikan struktur proyek menjadi beberapa file.
2. Tambahkan `.dockerignore`.
3. Tambahkan README berisi cara menjalankan lokal, Docker, dan konfigurasi secret.

## Ide Fitur yang Bisa Ditambahkan

### 1. Riwayat Analisis

Simpan hasil analisis sebelumnya dalam session atau database ringan:

- gambar,
- prediksi,
- confidence,
- tanggal,
- catatan pengguna,
- saran AI.

Manfaat: pengguna bisa membandingkan kondisi kulit kucing dari waktu ke waktu.

### 2. Mode Monitoring Perkembangan Luka/Penyakit

Pengguna upload foto hari ke-1, hari ke-3, hari ke-7, lalu aplikasi menampilkan:

- perubahan visual,
- confidence tiap waktu,
- catatan membaik/memburuk,
- rekomendasi kapan harus ke dokter.

### 3. Form Gejala Tambahan

Tambahkan pertanyaan singkat:

- kucing sering menggaruk?
- ada bulu rontok?
- ada kerak/luka?
- menular ke kucing lain?
- sudah berapa hari?

Hasil model gambar bisa digabung dengan gejala untuk membuat saran lebih relevan.

### 4. Rekomendasi Tindakan Aman Berbasis Confidence

Contoh:

- Confidence tinggi: tampilkan penjelasan dan saran dokter.
- Confidence sedang: sarankan upload ulang dengan pencahayaan lebih baik.
- Confidence rendah: jangan tampilkan diagnosis, minta foto ulang.

Ini membuat aplikasi terasa lebih bertanggung jawab.

### 5. Panduan Foto yang Baik

Sebelum upload, tampilkan contoh/panduan:

- area kulit terlihat jelas,
- pencahayaan cukup,
- tidak blur,
- jarak tidak terlalu jauh,
- hindari filter,
- sertakan foto close-up dan foto tubuh.

Manfaat: meningkatkan kualitas input dan akurasi prediksi.

### 6. Deteksi Kualitas Foto

Tambahkan pengecekan otomatis:

- blur,
- terlalu gelap,
- terlalu terang,
- resolusi terlalu kecil,
- objek terlalu jauh.

Jika kualitas buruk, aplikasi meminta upload ulang sebelum menjalankan model penyakit.

### 7. Export PDF Hasil Analisis

Buat tombol untuk mengunduh laporan:

- prediksi,
- probabilitas,
- Grad-CAM,
- saran AI,
- disclaimer,
- tanggal pemeriksaan.

Manfaat: pengguna bisa membawa hasil awal ke dokter hewan.

### 8. Direktori Klinik Hewan yang Lebih Terarah

Selain Google Maps, aplikasi bisa menampilkan:

- klinik terdekat berdasarkan lokasi,
- nomor telepon,
- jam buka,
- rating,
- tombol rute.

Untuk ini, bisa memakai Google Places API jika ingin lebih akurat.

### 9. Multi-Bahasa

Tambahkan pilihan bahasa:

- Indonesia,
- Inggris,
- bahasa daerah jika target pengguna lokal.

### 10. Admin Dashboard Evaluasi Model

Tambahkan dashboard untuk developer:

- distribusi prediksi,
- rata-rata confidence,
- jumlah upload gagal,
- kelas yang paling sering muncul,
- contoh prediksi confidence rendah.

Manfaat: membantu meningkatkan dataset dan model.

### 11. Feedback Pengguna

Setelah prediksi, pengguna bisa memilih:

- benar,
- salah,
- tidak yakin,
- sudah dikonfirmasi dokter.

Data ini bisa dipakai untuk evaluasi model berikutnya.

### 12. Mode Edukasi Penyakit Kulit Kucing

Buat halaman informasi untuk:

- Flea Allergy,
- Ringworm,
- Scabies,
- kulit sehat.

Isi bisa berupa gejala umum, risiko penularan, pencegahan, dan kapan harus ke dokter.

### 13. Peringatan Penyakit Menular

Untuk Ringworm dan Scabies, tampilkan peringatan bahwa penyakit bisa menular ke hewan lain dan pada beberapa kasus ke manusia. Sarankan isolasi sementara dan konsultasi dokter hewan.

### 14. Model Ensemble atau Two-Stage Classifier

Arsitektur yang lebih kuat:

1. Model pertama: validasi gambar kucing/kulit/bukan kucing.
2. Model kedua: klasifikasi penyakit kulit.

Manfaat: mengurangi prediksi percaya diri pada gambar yang tidak relevan.

### 15. Integrasi Chat Follow-Up

Setelah hasil keluar, pengguna bisa bertanya:

- "Apakah ini menular?"
- "Apa yang harus saya lakukan malam ini?"
- "Kapan harus ke dokter?"

Chat harus tetap diberi batasan medis agar tidak memberi diagnosis pasti atau dosis obat.

## Saran Struktur Kode Baru

Saat ini semua logic berada di `app.py`. Untuk proyek yang akan dikembangkan, lebih rapi jika dipisah:

```txt
app.py
src/
  config.py
  model.py
  preprocessing.py
  validation.py
  gradcam.py
  openrouter.py
  maps.py
tests/
  test_preprocessing.py
  test_model_loading.py
```

Manfaat:

- lebih mudah dites,
- lebih mudah debug,
- fitur baru tidak membuat `app.py` semakin panjang,
- model dan UI tidak bercampur terlalu kuat.

## Kesimpulan

Kode FelineSkin.AI sudah memiliki fondasi yang bagus untuk demo/prototype: upload gambar, klasifikasi penyakit, Grad-CAM, saran AI, dan pencarian dokter. Checkpoint model juga cocok dengan jumlah kelas saat ini.

Namun, sebelum dipakai lebih serius, ada beberapa hal penting yang perlu dibenahi: error handling, validasi API key, stabilitas load model, akurasi validasi gambar kucing, Grad-CAM hook, warna overlay, dan dependency pinning. Setelah itu, fitur paling bernilai untuk ditambahkan adalah riwayat analisis, form gejala tambahan, deteksi kualitas foto, export PDF, dan feedback pengguna untuk memperbaiki model di versi berikutnya.
