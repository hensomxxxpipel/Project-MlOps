import pandas as pd
import mlflow
from pathlib import Path
from google_play_scraper import reviews_all, Sort

mlflow.set_experiment("Projek MlOps")

APP_ID = 'id.ac.ub.gapura_mobile'
countries = ['id', 'us', 'sg', 'my', 'au']

# Tentukan folder data relatif dari src/
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

output_file = DATA_DIR / "data_mentah_review_gapura.csv"

all_reviews = []

with mlflow.start_run(run_name="scraper_gapura"):
    for country in countries:
        try:
            reviews = reviews_all(
                APP_ID,
                lang='id',
                country=country,
                sort=Sort.NEWEST,
                sleep_milliseconds=0
            )
            print(f"Selesai mengambil dari {country}. Jumlah review: {len(reviews)}")
            all_reviews.extend(reviews)

            mlflow.log_metric(f"reviews_{country}", len(reviews))
        except Exception as e:
            print(f"Terjadi kesalahan saat scraping {country}: {e}")
            mlflow.log_param(f"error_{country}", str(e))

    # Ubah ke DataFrame
    df = pd.DataFrame(all_reviews)
    df = df.drop_duplicates(subset=['reviewId'])

    # Simpan ke CSV
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"Total review unik yang disimpan: {len(df)}")

    # Log ke MLflow
    mlflow.log_metric("total_reviews", len(df))
    mlflow.log_param("output_file", str(output_file))

    # Simpan dataset ke MLflow (artifact)
    mlflow.log_artifact(str(output_file), artifact_path="datasets")
