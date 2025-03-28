from flask import Flask, request, jsonify

app = Flask(_name_)


@app.route('/')
def home():
    return jsonify({"message": "Sound Classification API is running!"})


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    return jsonify({"message": "File received successfully!"})


if _name_ == '_main_':
    app.run(debug=True, host="0.0.0.0", port=5000)