import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from pathlib import Path

# Path dataset
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INPUT_FILE = DATA_DIR / "review_gapura_stemming.csv"

# Load data hasil preprocessing
df = pd.read_csv(INPUT_FILE)

# Pastikan kolom yang dibutuhkan ada
if "sentiment" not in df.columns or "cleaned" not in df.columns:
    raise ValueError("Kolom 'sentiment' dan 'cleaned' harus ada di CSV!")

# Buat word cloud untuk setiap label sentimen
sentiments = df["sentiment"].unique()

for sentiment in sentiments:
    text = " ".join(df[df["sentiment"] == sentiment]["cleaned"].astype(str))
    
    wordcloud = WordCloud(
        width=800, height=400,
        background_color="white",
        max_words=200,
        collocations=False
    ).generate(text)

    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Word Cloud untuk Sentimen: {sentiment}", fontsize=16)
    plt.show()
