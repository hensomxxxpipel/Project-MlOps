# Projek MLOps - Analisis Sentimen Review Gapura

Proyek ini bertujuan untuk membangun pipeline **MLOps sederhana** untuk analisis sentimen review aplikasi Gapura.
Proses meliputi scraping data, labeling sentimen, preprocessing teks, balancing dataset, training model dengan MLflow, serta visualisasi word cloud.

---

## 📂 Struktur Direktori

```
projek-mlops/
│
├── data/                                   # Semua dataset
│   ├── data_mentah_review_gapura.csv       # Output scraper.py → Input labeling.py
│   ├── review_gapura_dengan_sentiment.csv  # Output labeling.py → Input preprocessing.py
│   ├── review_gapura_checkpoint.csv        # Output labeling sementara
│   ├── review_gapura_stemming_balanced.csv # Output preprocessing.py → Input training.py
│   └── review_gapura_stemming.csv          # Output preprocessing.py → Input word_cloud.py
│
├── model/                                  # Model hasil training
│   ├── svm_model.pkl
│   ├── naive_bayes_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── src/                                    # Semua source code
│   ├── scraper.py                          # Script scraping review
│   ├── labeling.py                         # Script memberi label sentimen
│   ├── preprocessing.py                    # Script preprocessing & balancing
│   ├── word_cloud.py                       # Script generate wordcloud
│   └── training.py                         # Script training + logging MLflow
│
├── .env                                    # Konfigurasi env (API_KEY, dsb.)
├── requirements.txt                        # Dependency project
└── README.md                               # Dokumentasi project
```

---

## 🚀 Setup Environment

1. **Clone repo**

   ```bash
   git clone https://github.com/username/projek-mlops.git
   cd projek-mlops
   ```

2. **Buat virtual environment & install dependencies**

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows

   pip install -r requirements.txt
   ```

3. **Konfigurasi `.env`**
   Buat file `.env` di root project. Contoh:

   ```
   API_KEY=xxxxxx 
   ```
   Buka website google gemini untuk mendapatkann API Key

---

## ⚙️ Alur Pipeline

1. **Scraping Review**

   ```bash
   python src/scraper.py
   ```

   Output: `data/data_mentah_review_gapura.csv`

2. **Labeling Sentimen**

   ```bash
   python src/labeling.py
   ```

   Output: `data/review_gapura_dengan_sentiment.csv`

3. **Preprocessing & Balancing**

   ```bash
   python src/preprocessing.py
   ```

   Output:

   * `data/review_gapura_stemming.csv`
   * `data/review_gapura_stemming_balanced.csv`

4. **Word Cloud**

   ```bash
   python src/word_cloud.py
   ```

   Output: visualisasi word cloud

5. **Training & Logging dengan MLflow**

   ```bash
   python src/training.py
   ```

   Output:

   * Model: `model/svm_model.pkl`, `model/naive_bayes_model.pkl`, `model/tfidf_vectorizer.pkl`
   * MLflow tracking experiment: `Projek MlOps`

---

## 📊 MLflow Tracking

Untuk melihat hasil training:

```bash
mlflow ui
```

Buka di browser: [http://localhost:5000](http://localhost:5000)

---

## 📌 Catatan

* Dataset mentah hasil scraping perlu dilakukan labeling dengan Gemini 2.5 Flash-Lite (Proses +-20 menit).
* Word cloud tidak melalui proses balancing dataset agar lebih natural.
* Model yang digunakan: **SVM** dan **Naive Bayes** dengan **TF-IDF** sebagai representasi teks.

---
