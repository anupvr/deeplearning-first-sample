import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
import joblib  # ✅ Use joblib instead of pickle

# 1. Load dataset
df = pd.read_csv("spam.csv", encoding='latin-1')[["v1", "v2"]]
df.columns = ["label", "text"]
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# 2. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# 3. Vectorize text
vectorizer = TfidfVectorizer(max_features=1000)
X_train_vec = vectorizer.fit_transform(X_train).toarray()  # ✅ fit() happens here
X_test_vec = vectorizer.transform(X_test).toarray()

# ✅ Confirm vectorizer is fitted
print("Vectorizer fitted. IDF shape:", vectorizer.idf_.shape)

# 4. Define model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, input_shape=(1000,), activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# 5. Train model
model.fit(X_train_vec, y_train, epochs=5, batch_size=32, validation_split=0.1)

# 6. Evaluate
loss, acc = model.evaluate(X_test_vec, y_test)
print(f"Test Accuracy: {acc:.4f}")

# 7. Save model and fitted vectorizer
model.save("spam_model.h5")
joblib.dump(vectorizer, "vectorizer.pkl")  # ✅ safe save
print("Vectorizer fitted. IDF shape:", vectorizer.idf_.shape)
print("Model and vectorizer saved successfully.")