from flask import Flask, jsonify, request
from urllib.request import urlretrieve
import cv2
from deepface import DeepFace

app = Flask(__name__)

@app.route('/detect-emotion', methods=['POST'])
def detect_emotion():
    try:
        # Get image URL from request data
        image_url = request.json.get('image_url')
        
        if not image_url:
            raise ValueError('No image_url provided')
        
        # Download image and save it locally
        image_path, _ = urlretrieve(image_url)
        
        # Read image using OpenCV
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError('Failed to read image')
        
        # Detect emotions using Deepface
        emotions = DeepFace.analyze(image, actions=['emotion'])
        
        # Return detected emotions as a JSON response
        return jsonify(emotions[0]['dominant_emotion']), 200
    
    except ValueError as ve:
        # Handle specific value errors
        return jsonify({'error': str(ve)}), 400
    
    except Exception as e:
        # Handle other exceptions
        return jsonify({'error': 'An error occurred'}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')

