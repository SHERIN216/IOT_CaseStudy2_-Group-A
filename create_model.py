import face_recognition
import os
import pickle

known_faces_dir = "known_faces"
face_encodings = []
face_names = []

for filename in os.listdir(known_faces_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            face_encodings.append(encodings[0])
            face_names.append(os.path.splitext(filename)[0])
            print(f"✅ Encoded: {filename}")
        else:
            print(f"⚠️ No face found in {filename}, skipping.")

# Save to pickle file
model_data = {"encodings": face_encodings, "names": face_names}
with open("face_model.pkl", "wb") as f:
    pickle.dump(model_data, f)

print("🎉 Face model saved as face_model.pkl")
