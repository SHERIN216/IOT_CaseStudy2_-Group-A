#  Face Recognition Alert System (Raspberry Pi Ready)

This project uses a webcam and face recognition to **identify people at the door**. It announces the names of recognized individuals or warns when someone is unrecognized. The script is designed to work on any system including **Raspberry Pi**.

---

## 📁 Folder Structure
face_recognition
    main.py
known_faces
    vandit.jpg
    requirement.txt 
    README.md    



- `main.py`: The core script that runs face recognition and announcements.
- `known_faces/`: Folder containing images of people to recognize.
- `requirements.txt`: Python packages needed.
- `README.md`: You're reading it!

---

## 🚀 Setup Instructions

### 1. Install Python Packages

Open a terminal and run:

inside the above folder 
pip install -r requirements.txt

 optional if pyttsx3 fails to speak 
for raspberry pi 
sudo apt install espeak


2. Add Known Faces
Put clear front-facing images of people into the known_faces/ folder.

File names become the names spoken.

Example: john.jpg → "John recognized at the door."

run the script 
python3 main.py



