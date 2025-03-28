import face_recognition
import cv2
import os
import time

# Try pyttsx3 or fallback to espeak
try:
    import pyttsx3
    tts_engine = pyttsx3.init()
    def speak(text):
        tts_engine.say(text)
        tts_engine.runAndWait()
except:
    def speak(text):
        os.system(f'espeak "{text}"')

# Load known faces
known_face_encodings = []
known_face_names = []

known_faces_dir = "known_faces"
for filename in os.listdir(known_faces_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(os.path.splitext(filename)[0])
            print(f"Loaded face: {filename}")
        else:
            print(f" No face found in {filename}, skipping.")

video_capture = cv2.VideoCapture(0)
print("\n🎥 Camera is on. Press Ctrl+C to stop.\n")

try:
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print(" Could not access the camera.")
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)

            if True in matches:
                name = known_face_names[matches.index(True)]
                print(f" Recognized: {name}")
                speak(f"{name} recognized at the door.")
            else:
                print(" Unrecognized person detected!")
                speak("Unrecognized person at the door. Proceed with caution.")

        time.sleep(1)

except KeyboardInterrupt:
    print("\n Program stopped by user.")
finally:
    video_capture.release()
    print(" Camera released.")
