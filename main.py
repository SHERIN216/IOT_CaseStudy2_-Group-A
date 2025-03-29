import face_recognition
import cv2
import pickle
import os
import time

# Voice setup
try:
    import pyttsx3
    tts_engine = pyttsx3.init()
    def speak(text):
        tts_engine.say(text)
        tts_engine.runAndWait()
except:
    def speak(text):
        os.system(f'espeak "{text}"')

# Load face model
with open("face_model.pkl", "rb") as f:
    model_data = pickle.load(f)
known_face_encodings = model_data["encodings"]
known_face_names = model_data["names"]

print(f"✅ Loaded model with {len(known_face_names)} faces.")

video_capture = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("⚠️ Camera error.")
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)

            if True in matches:
                name = known_face_names[matches.index(True)]
                print(f"✅ Recognized: {name}")
                speak(f"{name} recognized at the door.")
            else:
                print("🚨 Unknown person detected!")
                speak("Unrecognized person at the door. Proceed with caution.")

        time.sleep(1)

except KeyboardInterrupt:
    print("\n🛑 Stopped by user.")
finally:
    video_capture.release()
    print("🎬 Camera released.")
