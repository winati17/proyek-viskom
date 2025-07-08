# **Tugas-Akhir-Viskom**

**Nama** : Winati Mutmainnah
**NIM** : D121221014

## Tentang Kode Program

Program ini bertujuan untuk melakukan eksperimen penambahan noise dan filtering (denoising) citra digital menggunakan Python. Fokus utama program adalah membandingkan efektivitas dua metode filter: Gaussian Filter dan Median Filter terhadap tiga jenis noise:

- Gaussian Noise
- Salt & Pepper Noise
- Speckle Noise

Analisis dilakukan dengan mengukur metrik kualitas citra hasil filtering terhadap citra asli menggunakan PSNR (Peak Signal-to-Noise Ratio), MSE (Mean Squared Error), dan SSIM (Structural Similarity Index).

Selain itu, program ini juga membandingkan hasil filtering pada berbagai kernel size dan nilai sigma untuk melihat pengaruh parameter terhadap performa filter.

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
2. Install dependensi dengan menambah sel %pip install numpy matplotlib pillow
3. Sesuaikan path citra input  
4. Jalankan program
