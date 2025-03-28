import librosa
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Function to extract MFCC features from an audio file
def extract_features(audio_file):
    try:
        # Load the audio file
        audio, sr = librosa.load(audio_file, sr=None)
        # Extract 13 MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        # Take the mean across the time axis
        mfcc = np.mean(mfcc, axis=1)
        return mfcc
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return None  # Return None if file cannot be processed

# Function to load audio data from a CSV file
def load_data_from_csv(csv_path):
    df = pd.read_csv(csv_path)

    # Check if required columns exist
    if 'file_path' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV file must contain 'file_path' and 'label' columns.")

    features, labels = [], []

    for _, row in df.iterrows():
        audio_file = row['file_path']
        label = row['label']
        mfcc = extract_features(audio_file)

        if mfcc is not None:
            features.append(mfcc)
            labels.append(label)

    return np.array(features), np.array(labels)

# Load dataset
csv_file_path = 'UrbanSound8K.csv'  # Change to your CSV file path
X, y = load_data_from_csv(csv_file_path)

# Split into train-test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model using pickle
model_filename = "sound_classifier.pkl"
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)

print(f"Model saved as {model_filename}")