#!/usr/bin/env python3
import numpy as np
import sounddevice as sd
import cv2
from scipy.io import wavfile
import tensorflow.lite as tflite
from flask import Flask, request, jsonify
import threading
import time
import os
import pyttsx3
import RPi.GPIO as GPIO


# ======================
# Hardware Setup
# ======================
# GPIO.setmode(GPIO.BCM)
# ALARM_PIN = 17  # GPIO pin for siren
# GPIO.setup(ALARM_PIN, GPIO.OUT)

# ======================
# Configuration
# ======================
# AUDIO_SAMPLE_RATE = 44100
# SOUND_THRESHOLDS = {
#    'fire_alarm': {'min_freq': 3000, 'max_amplitude': 0.8},
#    'glass_break': {'min_freq': 5000, 'max_amplitude': 0.6}
# }

# ======================
# Core Functions
# ======================
def classify_audio(audio_data):
    """Rule-based sound classification (replace with ML model if available)"""
    fft = np.fft.fft(audio_data)
    freqs = np.fft.fftfreq(len(fft), 1 / AUDIO_SAMPLE_RATE)
    dominant_freq = np.abs(freqs[np.argmax(np.abs(fft))])
    amplitude = np.max(np.abs(audio_data))

    if (SOUND_THRESHOLDS['fire_alarm']['min_freq'] <= dominant_freq <= 4000
            and amplitude > SOUND_THRESHOLDS['fire_alarm']['max_amplitude']):
        return "fire_alarm"
    elif dominant_freq >= SOUND_THRESHOLDS['glass_break']['min_freq']:
        return "glass_break"
    return "normal"


def detect_visual_events(frame):
    """Face mask detection using Haar cascades"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return len(faces) > 0  # Simplified: Just detect faces


# ======================
# Decision Engine
# ======================
EVENT_RESPONSES = {
    'fire_alarm': {
        'message': "Fire detected! Evacuate immediately!",
        'actions': ['announce', 'trigger_alarm']
    },
    'glass_break': {
        'message': "Glass break detected! Possible intrusion!",
        'actions': ['announce', 'alert_security']
    },
    'face_detected': {
        'message': "Unauthorized person detected!",
        'actions': ['announce']
    }
}


def execute_action(action, message):
    if action == 'announce':
        engine = pyttsx3.init()
        engine.say(message)
        engine.runAndWait()
    elif action == 'trigger_alarm':
        GPIO.output(ALARM_PIN, GPIO.HIGH)
        time.sleep(5)
        GPIO.output(ALARM_PIN, GPIO.LOW)


# ======================
# Flask API & Main Loop
# ======================
app = Flask(__name__)


@app.route('/audio_upload', methods=['POST'])
def audio_upload():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'}), 400

    audio_file = request.files['audio']
    temp_path = f"temp_audio.wav"
    audio_file.save(temp_path)
    _, audio_data = wavfile.read(temp_path)
    os.remove(temp_path)

    event = classify_audio(audio_data)
    if event != 'normal':
        handle_event(event)
    return jsonify({'status': 'processed', 'event': event})


def camera_monitor():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if detect_visual_events(frame):
            handle_event('face_detected')
        time.sleep(1)


def handle_event(event_type):
    response = EVENT_RESPONSES.get(event_type)
    if response:
        for action in response['actions']:
            execute_action(action, response['message'])


if __name__ == '__main__':
    # Start camera thread
    threading.Thread(target=camera_monitor, daemon=True).start()

    # Start Flask server
    app.run(host='0.0.0.0', port=5000)