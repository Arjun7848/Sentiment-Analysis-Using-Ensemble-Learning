import pandas as pd
import joblib
import matplotlib.pyplot as plt

from preprocess import clean_text

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from scipy.sparse import hstack


# ==========================================
# 1. LOAD DATASET
# ==========================================

columns = ["Tweet_ID", "Entity", "Sentiment", "Tweet"]

data = pd.read_csv(
    "dataset/twitter_training.csv",
    header=None,
    names=columns
)

print("Dataset loaded successfully!")
print("Original rows:", len(data))


# ==========================================
# 2. KEEP ONLY 3 SENTIMENT CLASSES
# ==========================================

data = data[
    data["Sentiment"].isin(["Positive", "Negative", "Neutral"])
]

data = data.dropna(subset=["Tweet"])

print("Rows after filtering:", len(data))

print("\nSentiment distribution:")
print(data["Sentiment"].value_counts())


# ==========================================
# 3. TEXT PREPROCESSING
# ==========================================

print("\nCleaning tweets...")

data["Clean_Tweet"] = data["Tweet"].apply(clean_text)

print("Text preprocessing completed!")


# ==========================================
# 4. WORD TF-IDF
# ==========================================

print("\nCreating Word TF-IDF...")

word_vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

X_word = word_vectorizer.fit_transform(
    data["Clean_Tweet"]
)

print("Word TF-IDF shape:", X_word.shape)


# ==========================================
# 5. CHARACTER TF-IDF
# ==========================================

print("\nCreating Character TF-IDF...")

char_vectorizer = TfidfVectorizer(
    analyzer="char",
    max_features=5000,
    ngram_range=(3, 5),
    min_df=2,
    sublinear_tf=True
)

X_char = char_vectorizer.fit_transform(
    data["Clean_Tweet"]
)

print("Character TF-IDF shape:", X_char.shape)


# ==========================================
# 6. COMBINE FEATURES
# ==========================================

print("\nCombining TF-IDF features...")

X = hstack([X_word, X_char])

y = data["Sentiment"]

print("Final feature shape:", X.shape)


# ==========================================
# 7. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain-Test Split completed!")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ==========================================
# 8. RANDOM FOREST
# ==========================================

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=150,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Random Forest training completed!")


# ==========================================
# 9. PREDICTION
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 10. EVALUATION
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n====================================")
print("MODEL PERFORMANCE")
print("====================================")

print(
    "Accuracy: {:.2f}%".format(
        accuracy * 100
    )
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)


# ==========================================
# 11. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

display.plot()

plt.title(
    "Random Forest - 3 Class Sentiment Analysis"
)

plt.tight_layout()

plt.savefig(
    "models/confusion_matrix_3class.png"
)

plt.close()

print("\nConfusion matrix saved!")


# ==========================================
# 12. SAVE MODEL
# ==========================================

joblib.dump(
    model,
    "models/sentiment_model_3class.pkl"
)

joblib.dump(
    word_vectorizer,
    "models/word_tfidf_vectorizer.pkl"
)

joblib.dump(
    char_vectorizer,
    "models/char_tfidf_vectorizer.pkl"
)

print("Model saved!")
print("Word vectorizer saved!")
print("Character vectorizer saved!")


print("\n====================================")
print("3-CLASS TRAINING COMPLETED!")
print("====================================")