import os
import time
import re
import pandas as pd
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# model Gemini 2.5
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

# prompt template
prompt = PromptTemplate.from_template("""
Klasifikasikan teks review berikut ke dalam salah satu kategori:
- positif
- negatif
- netral
Balas hanya dengan label (positif/negatif/netral).

Teks: {review}
Label:
""")

parser = StrOutputParser()
chain = prompt | llm | parser

# Path file input/output
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

input_file = DATA_DIR / "data_mentah_review_gapura.csv"
checkpoint_file = DATA_DIR / "review_gapura_checkpoint.csv"
output_file = DATA_DIR / "review_gapura_dengan_sentiment.csv"

# load data
df = pd.read_csv(input_file)
df_selected = df[["reviewId", "content", "score", "at"]].copy()
df_selected["sentiment"] = None 

# batasi request: 15 per menit
requests_per_minute = 15
counter = 0

for i in range(len(df_selected)):
    text = str(df_selected.at[i, "content"])
    sentiment = "netral" if pd.isnull(text) else None

    if sentiment is None:
        while True:
            try:
                sentiment = chain.invoke({"review": text}).strip()
                break
            except Exception as e:
                err_msg = str(e)
                if "ResourceExhausted" in err_msg or "429" in err_msg:
                    wait_time = 60
                    match = re.search(r"retry in (\d+(\.\d+)?)s", err_msg)
                    if match:
                        wait_time = float(match.group(1)) + 2
                    print(f"Rate limit/Quota exceeded. Tunggu {wait_time} detik lalu coba ulang...")
                    time.sleep(wait_time)
                    continue
                else:
                    sentiment = f"ERROR: {e}"
                    break

    df_selected.at[i, "sentiment"] = sentiment
    counter += 1

    # tiap 15 request istirahat 1 menit
    if counter % requests_per_minute == 0:
        print(f"Sudah {counter} request, istirahat 60 detik...")
        time.sleep(60)

    # autosave tiap 100 baris
    if counter % 100 == 0:
        df_selected.to_csv(checkpoint_file, index=False, encoding="utf-8-sig")
        print(f"Checkpoint disimpan ({counter} data).")

# simpan hasil akhir
df_selected.to_csv(output_file, index=False, encoding="utf-8-sig")
print(f"Selesai. Total {len(df_selected)} data tersimpan.")
print(df_selected.head())
