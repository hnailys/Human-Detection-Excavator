# Human-Detection-Excavator
Sistem deteksi manusia di area belakang ekskavator menggunakan YOLOv5 untuk meningkatkan keselamatan kerja dengan mendeteksi manusia di zona bahaya dan memberikan peringatan waktu nyata kepada operator.

# Deteksi Manusia untuk Ekskavator menggunakan YOLOv5
Proyek ini bertujuan untuk mengembangkan sistem deteksi manusia di area belakang ekskavator menggunakan model deteksi objek YOLOv5. Sistem ini menganalisis video dari kamera yang dipasang di ekskavator untuk mendeteksi keberadaan manusia dan mengukur jarak mereka dari ekskavator. Hasil deteksi termasuk informasi visual pada video dan panel samping yang menunjukkan jumlah manusia yang terdeteksi dan jumlah deteksi jarak dekat (kurang dari 3 meter).

## Persyaratan
- Python 3.x
- OpenCV
- NumPy

## Persiapan
1. Clone repository ini:
    ```bash
    git clone https://github.com/hnailys/Human-Detection-Excavator.git
    cd Human-Detection-Excavator
    ```

2. Instal pustaka yang diperlukan:
    ```bash
    pip install opencv-python numpy
    ```

3. Tempatkan file video Anda di folder `ReferenceVideos` dan perbarui jalur video di `detect_humans.py` sesuai.

4. Unduh model YOLOv5 dan letakkan file `best.onnx` di direktori proyek.

## Menjalankan Deteksi
Jalankan perintah berikut untuk memulai deteksi:
```bash
python detect_humans.py
