# Abstrak
	Rokok termasuk zat adiktif karena dapat menyebabkan ketagihan yang berdampak 
sulit untuk berhenti dan menjadi salah satu penyebab kematian utama di dunia.Banyak 
jenis penyakit yang bisa mengancam kesehatan akibat rokok pada perokok aktif atau 
perokok pasif dan juga bisa mengganggu masyarakat lainnya saat melakukan aktivitas 
pola hidup sehat terutama di kawasan tanpa asap rokok. Hal ini lah yang menjadi 
motivasi dasar pada penelitian ini untuk membangun model penerapan deep learning
pada klasifikasi citra untuk mendeteksi orang merokok dan tidak merokok, berbasis
website agar dapat digunakan untuk melakukan monitoring pada kawasan tanpa asap rokok. 
	Metode penelitian yang digunakan dalam penelitian ini memanfaatkan metode 
Convolutional Neural Network dengan arsitektur MobileNetV2 dengan menggunakan dataset 
berupa gambar merokok dan tidak merokok, kemudian diolah melalui tahap preprocessing 
yaitu berupa augmentasi data, pemodelan dan implementasi.
	Penelitian ini dibangun berbasis website menggunakan bahasa pemrograman 
Python dengan framework flask untuk merancang interface. Aplikasi yang dibangun 
memiliki fitur yang menampilkan tentang penelitian, penjelasan model, dataset,
dan halaman testing dengan menggunakan real time webcam atau input file. Hasil 
dari pengujian akan menunjukkan nilai akurasi 92.30%, presisi sebesar 95.65%, 
dan recall sebesar 88%. Hal ini menunjukkan sistem layak untuk digunakan.

## Framework
-  Tensorflow
-  MobilenetV2
-  Flask
-  OpenCV
-  Matplotlib

## Cara running aplikasi
1. Lakukan instalasi packages dengan menggunakan 2 cara ini:
- menggunakan pip
```pip install -r requirements.txt```
- menggunakan conda
```conda env create -f environment.yml```

setelah packages terinstal, masukan command di bawah pada root folder:

```
python wsgi.py
```

## Data
Dataset dapat didownload pada website resmi kaggle  <a href="https://www.kaggle.com/code/raj713335/cigarette-smoker-detection/data">disini</a>.

##Referensi
https://github.com/GalileoParise/CV-Mask-detection
