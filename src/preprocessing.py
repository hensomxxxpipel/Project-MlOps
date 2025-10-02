import pandas as pd
import re
from pathlib import Path
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.utils import resample

# Path data
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INPUT_FILE = DATA_DIR / "review_gapura_dengan_sentiment.csv"
OUTPUT_FILE = DATA_DIR / "review_gapura_stemming.csv"
OUTPUT_BALANCED_FILE = DATA_DIR / "review_gapura_stemming_balanced.csv"

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', ' ', str(text))
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Baca dataset
df = pd.read_csv(INPUT_FILE)

# Cleaning
df["cleaned"] = df["content"].astype(str).apply(clean_text)
df = df.dropna(subset=["cleaned"])
df = df.drop_duplicates(subset=["reviewId"], keep="first")

# Stopword removal + stemming
stemmer = StemmerFactory().create_stemmer()
stop_factory = StopWordRemoverFactory()
stop_remover = stop_factory.create_stop_word_remover()

def stem_stopword(text):
    text = stop_remover.remove(text)
    return stemmer.stem(text)

df["stemming"] = df["cleaned"].apply(stem_stopword)

# Simpan hasil tanpa sampling
df.to_csv(OUTPUT_FILE, index=False)

# Oversampling agar seimbang
max_count = 149
df_balanced = pd.DataFrame()

for label in df["sentiment"].unique():
    subset = df[df["sentiment"] == label]
    subset_resampled = resample(
        subset,
        replace=True,
        n_samples=max_count,
        random_state=42
    )
    df_balanced = pd.concat([df_balanced, subset_resampled])

df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Simpan hasil balanced
df_balanced.to_csv(OUTPUT_BALANCED_FILE, index=False)

print("Preprocessing selesai!")
print(f"Hasil disimpan")
print(f"Hasil balanced juga disimpan")