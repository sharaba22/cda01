"""
ml_model_improved.py
--------------------
Improved machine learning pipeline for Darknet.csv

Fixes:
- Handles inf / NaN / extremely large values before scaling
- Encodes all categorical features automatically
- Scales features safely
- Builds an optimized neural network with dropout & early stopping
- Automatically adjusts output layer for the number of classes
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras import layers, models, callbacks
from collections import Counter


# Load Dataset
print("Loading dataset...")
df = pd.read_csv("Darknet.csv")
print("Original shape:", df.shape)

# Drop unnecessary columns if they exist
df = df.drop(columns=["Flow ID", "Timestamp"], errors="ignore")


# Encode categorical columns
print("Encoding categorical columns...")
for c in df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df[c] = le.fit_transform(df[c].astype(str))

print("All object columns encoded.")


# Handle numeric issues (inf, NaN, very large values)
print("Cleaning numeric data...")

# Replace inf/-inf with NaN
df = df.replace([np.inf, -np.inf], np.nan)

# Replace NaN with column median
df = df.fillna(df.median(numeric_only=True))

# Cap extreme outliers to prevent overflow (|x| > 1e12)
for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].clip(lower=-1e12, upper=1e12)

print("Data cleaned. No inf/NaN remain:",
      np.isinf(df.values).sum(), "inf; ",
      np.isnan(df.values).sum(), "NaN")

# Prepare data for training
target = "Label1"  # Task 1 target
if target not in df.columns:
    raise ValueError(f"Target column '{target}' not found in dataset!")

X = df.drop(columns=[target])
y = df[target]

print("Feature shape:", X.shape)
print("Label distribution:", Counter(y))

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Scale numeric data safely
print("Scaling features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Scaling done.")

# Build improved neural network
n_features = X_train.shape[1]
n_classes = len(np.unique(y_train))

print(f"Building NN model: {n_features} features → {n_classes} classes")

model = models.Sequential([
    layers.Input(shape=(n_features,)),
    layers.Dense(256, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(128, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(n_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Training
es = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

print("Training model...")
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=512,
    callbacks=[es],
    verbose=2
)

# Evaluation
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Final Test Accuracy: {acc:.3f}")
print(f"✅ Final Test Loss: {loss:.3f}")