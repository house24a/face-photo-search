from flask import Flask, request, jsonify
import requests
import numpy as np
import json
import os

app = Flask(__name__)

API_KEY = os.environ.get('hn-O6IAVGcpkjgMJAWaA-arbe1S7wfls')
API_SECRET = os.environ.get('DvEDFA4EfF6FKe71kvAtlwZ1XtEQ-Huj')

with open('face_vectors.json', 'r') as f:
    face_db = json.load(f)

def cosine_similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def extract_feature(image_file):
    url = "https://api-us.faceplusplus.com/facepp/v3/detect"
    files = {"image_file": image_file}
    data = {
        "api_key": API_KEY,
        "api_secret": API_SECRET,
        "return_landmark": 0,
        "return_attributes": ""
    }
    response = requests.post(url, data=data, files=files)
    faces = response.json().get("faces", [])
    if not faces:
        return None
    face_token = faces[0]["face_token"]

    search_url = "https://api-us.faceplusplus.com/facepp/v3/face/analyze"
    data = {
        "api_key": API_KEY,
        "api_secret": API_SECRET,
        "face_tokens": face_token,
        "return_attributes": "embedding"
    }
    vec_response = requests.post(search_url, data=data)
    embedding = vec_response.json().get("faces", [{}])[0].get("attributes", {}).get("embedding", {}).get("vector")
    return embedding

@app.route('/upload', methods=['POST'])
def upload():
    if 'selfie' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    selfie = request.files['selfie']
    user_vector = extract_feature(selfie)
    if not user_vector:
        return jsonify({'error': 'Face not found'}), 400

    results = []
    for photo in face_db:
        sim = cosine_similarity(user_vector, photo['face_vector'])
        results.append((photo['photo_id'], sim))

    top_matches = sorted(results, key=lambda x: x[1], reverse=True)[:5]
    return jsonify({'matches': top_matches})

if __name__ == '__main__':
    app.run(debug=True)
