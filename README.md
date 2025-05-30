# **Tugas-Akhir-Viskom**

**Nama** : Winati Mutmainnah
**NIM** : D121221014

## Tentang Kode Program

Program ini bertujuan untuk membandingkan efektivitas beberapa metode denoising citra menggunakan bahasa pemrograman Python dan library seperti OpenCV, NumPy, dan Scikit-Image. Beberapa metode yang diuji meliputi:

- Gaussian Smoothing  
- Median Filtering  
- Bilateral Filtering  
- Wavelet Denoising

Evaluasi dilakukan berdasarkan metrik kualitas gambar: PSNR, SSIM, MSE, dan SNR. Program ini digunakan untuk menganalisis dampak setiap metode terhadap kualitas visual dan kesetiaan terhadap citra asli.

## Dataset Citra Input
Citra input uji dapat ditemukan pada folder `images/`

## Cara Menjalankan Program (Google Colab)

1. Buka Google Colab: [https://colab.research.google.com](https://colab.research.google.com)  
2. Upload file Python atau salin seluruh kode program ke dalam sel di notebook  
3. Upload citra input ke dalam lingkungan kerja (content) menggunakan ikon 📁 di sidebar kiri  
4. Sesuaikan path citra input jika nama berbeda  
5. Jalankan sel program utama dengan menekan ▶️ atau tekan `Ctrl + Enter`

## Cara Menjalankan Program di Lokal

1. Clone atau salin kode program  
2. Install dependensi yang dibutuhkan  
3. Sesuaikan path citra input  
4. Jalankan program melalui terminal atau IDE

## Dependencies

Untuk Google Colab, sebagian besar dependensi sudah tersedia.  
Untuk lokal, jalankan perintah berikut untuk menginstal dependensi:

```bash
pip install opencv-python numpy scikit-image matplotlib pywavelets
```
