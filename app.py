from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load your CNN model
model = load_model('model/my_model.keras')
class_names = ['Cat', 'Dog']

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    img_path = None

    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            # Save uploaded file
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            img_path = filepath

            # --- Put the prediction code here ---
            try:
                # Load and resize the image
                img = image.load_img(filepath, target_size=(64, 64))
                img_array = image.img_to_array(img)

                # Scale pixel values
                img_array = img_array / 255.0

                # Add batch dimension
                img_array = np.expand_dims(img_array, axis=0)  # shape (1, 128, 128, 3)

                # Predict
                pred = model.predict(img_array)

                # Binary classification (sigmoid)
                class_idx = int(pred[0] > 0.5)
                result = class_names[class_idx]

            except Exception as e:
                result = f"Error during prediction: {str(e)}"

    return render_template('index.html', result=result, img_path=img_path)


if __name__ == '__main__':
    app.run(debug=True)