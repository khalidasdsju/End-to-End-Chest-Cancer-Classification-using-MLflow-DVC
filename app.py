import os
import io
import base64
import random
from flask import Flask, request, render_template, jsonify
from PIL import Image

# Set demo mode flag
DEMO_MODE = True
print("Running in DEMO MODE - no TensorFlow or model required")
print("This is a demonstration of the UI only")

# Mock the numpy and tensorflow imports for demo mode
class MockNumpy:
    @staticmethod
    def argmax(arr):
        return 0

    @staticmethod
    def expand_dims(arr, axis=0):
        return arr

np = MockNumpy()

class MockTF:
    class keras:
        class preprocessing:
            class image:
                @staticmethod
                def img_to_array(img):
                    return [0]

# Define mock TensorFlow class
tf = MockTF()

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# In demo mode, we don't need to load a real model
model = None
class_names = ['Benign', 'Malignant']
print(f"Available classes: {class_names}")

# Preprocess the image (simplified for demo mode)
def preprocess_image(image, target_size=(224, 224)):
    # Just resize the image in demo mode
    image = image.resize(target_size)
    return [[[0]]]  # Return a dummy array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Read the image
        img = Image.open(io.BytesIO(file.read())).convert('RGB')

        # Convert image to base64 for display
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # In demo mode, return a random prediction
        import random
        predicted_class = random.choice(class_names)
        confidence = random.uniform(70.0, 95.0)

        # Save the image for display
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_image.jpg')
        img.save(img_path)

        return jsonify({
            'success': True,
            'prediction': predicted_class,
            'confidence': confidence,
            'image': img_str,
            'demo_mode': True
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
