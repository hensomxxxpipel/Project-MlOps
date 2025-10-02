import pandas as pd
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib
from mlflow.models.signature import infer_signature

mlflow.set_experiment("Projek MlOps")

# Path dataset & model
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "model"

input_file = DATA_DIR / "review_gapura_stemming_balanced.csv"
df = pd.read_csv(input_file)

X = df["stemming"].astype(str)
y = df["sentiment"].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

with mlflow.start_run(run_name="training_gapura_sentiment"):

    # Model 1: SVM
    svm_model = SVC(kernel="linear", random_state=42)
    svm_model.fit(X_train_tfidf, y_train)

    y_pred_svm = svm_model.predict(X_test_tfidf)
    acc_svm = accuracy_score(y_test, y_pred_svm)

    print("=== Hasil SVM ===")
    print("Accuracy:", acc_svm)
    print(classification_report(y_test, y_pred_svm))

    mlflow.log_param("model_svm", "SVC-linear")
    mlflow.log_metric("accuracy_svm", acc_svm)

    signature_svm = infer_signature(X_train_tfidf.toarray(), y_train)
    mlflow.sklearn.log_model(
        svm_model,
        name="model_svm",
        signature=signature_svm,
        input_example=X_train_tfidf[:1].toarray()
    )

    # Simpan SVM ke file
    svm_path = MODEL_DIR / "svm_model.pkl"
    joblib.dump(svm_model, svm_path)
    print("Model SVM disimpan")

    # Model 2: Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)

    y_pred_nb = nb_model.predict(X_test_tfidf)
    acc_nb = accuracy_score(y_test, y_pred_nb)

    print("=== Hasil Naive Bayes ===")
    print("Accuracy:", acc_nb)
    print(classification_report(y_test, y_pred_nb))

    mlflow.log_param("model_nb", "MultinomialNB")
    mlflow.log_metric("accuracy_nb", acc_nb)

    signature_nb = infer_signature(X_train_tfidf.toarray(), y_train)
    mlflow.sklearn.log_model(
        nb_model,
        name="model_nb",
        signature=signature_nb,
        input_example=X_train_tfidf[:1].toarray()
    )

    # Simpan NB ke file
    nb_path = MODEL_DIR / "naive_bayes_model.pkl"
    joblib.dump(nb_model, nb_path)
    print("Model Naive Bayes disimpan")

    # Simpan vectorizer juga
    vectorizer_path = MODEL_DIR / "tfidf_vectorizer.pkl"
    joblib.dump(vectorizer, vectorizer_path)
    print("TF-IDF vectorizer disimpan")

    # log vectorizer ke MLflow
    mlflow.log_artifact(str(vectorizer_path), artifact_path="vectorizer")

    # log dataset ke MLflow
    mlflow.log_artifact(str(input_file), artifact_path="dataset")
