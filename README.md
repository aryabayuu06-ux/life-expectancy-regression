# Prediksi Harapan Hidup dengan Machine Learning

## Deskripsi Proyek
Proyek ini merupakan praktikum machine learning untuk menganalisis dataset harapan hidup (*Life Expectancy*) dan membangun model regresi untuk memprediksi nilai `Life expectancy`.

Tahapan yang dilakukan dalam notebook:
- Memuat dan memahami dataset, yang pada awalnya terdiri dari 2.938 baris dan 22 kolom.
- Melakukan Exploratory Data Analysis (EDA), termasuk pemeriksaan informasi dataset, statistik deskriptif, missing values, duplikasi, distribusi data, boxplot, dan korelasi.
- Menangani missing values dengan mode untuk kolom kategorikal dan median untuk kolom numerik.
- Mengidentifikasi dan menyaring outlier menggunakan metode Interquartile Range (IQR).
- Melakukan standardisasi pada kolom numerik serta encoding kolom kategorikal untuk analisis dan pemodelan.
- Membagi data menjadi data training dan testing dengan proporsi 80:20.
- Melatih dan membandingkan tiga algoritma regresi: Least Angle Regression (LARS), Linear Regression, dan Gradient Boosting Regressor.
- Mengevaluasi model menggunakan Mean Absolute Error (MAE), Mean Squared Error (MSE), dan R-squared (R²).
- Membuat visualisasi perbandingan model, membandingkan nilai aktual dengan prediksi, dan menampilkan feature importance untuk model Gradient Boosting.
- Menyimpan model Gradient Boosting ke `gbr_model.joblib` serta model dengan nilai R² tertinggi ke `best_life_expectancy_model.joblib`.

Berdasarkan output notebook saat ini, Linear Regression menghasilkan nilai R² tertinggi di antara ketiga model yang diuji.

| Model | MAE | MSE | R² |
|---|---:|---:|---:|
| LARS | 0.5330 | 0.4712 | 0.5155 |
| Linear Regression | 0.2242 | 0.1522 | 0.8435 |
| Gradient Boosting Regressor | 0.2500 | 0.1536 | 0.8421 |

**Catatan interpretasi:** pada notebook saat ini, kolom numerik termasuk `Life expectancy` distandardisasi sebelum pembagian data. Karena itu, nilai MAE dan MSE di atas berada pada skala target yang telah distandardisasi, bukan satuan tahun harapan hidup.

## Struktur Folder
Pastikan susunan file di repository seperti berikut agar lokasi dataset sesuai dengan kode di notebook:

```text
life-expectancy-regression/
├── README.md
├── requirements.txt
├── Life_Expectancy_data.ipynb
├── Life Expectancy Data.csv
├── gbr_model.joblib
└── best_life_expectancy_model.joblib
```

> Nama notebook boleh berbeda. Sesuaikan nama pada struktur folder dengan file notebook yang diunggah. Notebook saat ini membaca dataset dengan `pd.read_csv("Life Expectancy Data.csv")`, sehingga file CSV perlu berada di folder yang sama dengan notebook.

## Dataset
Dataset yang digunakan adalah `Life Expectancy Data.csv`, yang berisi data harapan hidup serta sejumlah indikator demografi, kesehatan, dan ekonomi dari berbagai negara. Dataset awal memiliki 2.938 baris dan 22 kolom. Target prediksi pada proyek ini adalah `Life expectancy`.

Sumber dataset: [Life Expectancy (WHO) - Kaggle](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who).

Jika dataset dibagikan kembali melalui repository, pastikan mengikuti lisensi dan ketentuan penggunaan dari sumber aslinya.

## Persiapan dan Menjalankan Notebook

1. Instal Python dan pastikan versinya kompatibel dengan library yang tercantum di `requirements.txt`.
2. Unduh atau clone repository ini ke komputer.
3. Buka terminal pada folder proyek.
4. Instal library yang dibutuhkan:

   ```bash
   pip install -r requirements.txt
   ```

5. Buka notebook `Life_Expectancy_data.ipynb` menggunakan Jupyter Notebook, JupyterLab, atau VS Code.
6. Pastikan file `Life Expectancy Data.csv` berada di folder yang sama dengan notebook.
7. Jalankan sel notebook secara berurutan dari atas ke bawah.

## Memuat Model yang Sudah Disimpan

Model Gradient Boosting disimpan pada file `gbr_model.joblib`, sedangkan model yang dipilih berdasarkan nilai R² tertinggi pada hasil evaluasi notebook disimpan pada `best_life_expectancy_model.joblib`.

Untuk memuat kembali model:

```python
import joblib

# Memuat model Gradient Boosting
gbr_model = joblib.load("gbr_model.joblib")

# Memuat model dengan R² tertinggi pada hasil evaluasi notebook
best_model = joblib.load("best_life_expectancy_model.joblib")
```

File model tersebut menyimpan estimator yang telah dilatih, bukan keseluruhan alur preprocessing dalam satu pipeline. Sebelum digunakan untuk prediksi data baru, fitur perlu diproses dengan cara yang konsisten dengan proses training, termasuk urutan kolom dan encoding kategorikal.

## Catatan Penting
- Hasil evaluasi menggambarkan performa model pada eksperimen dan pembagian data yang digunakan pada notebook ini; hasil tersebut tidak membuktikan hubungan sebab-akibat.
- Pada versi notebook saat ini, standardisasi dilakukan sebelum train-test split dan mencakup target `Life expectancy`. Untuk evaluasi yang lebih ketat, standardisasi sebaiknya dipelajari hanya dari data training, dan metrik sebaiknya dihitung pada skala target yang jelas agar dapat diinterpretasikan dalam satuan tahun.
- Model `best_life_expectancy_model.joblib` dipilih berdasarkan R² tertinggi pada output notebook saat ini, yaitu Linear Regression.
- File model `.joblib` sebaiknya hanya dimuat jika berasal dari sumber yang tepercaya.
- Untuk reproduksibilitas yang lebih baik, gunakan versi Python dan library yang konsisten dengan lingkungan saat model dilatih.

## Teknologi
- Python
- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- Joblib
- Jupyter Notebook
