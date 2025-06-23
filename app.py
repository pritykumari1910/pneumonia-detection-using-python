# from flask import Flask, render_template, request, redirect, url_for
# import os
# from werkzeug.utils import secure_filename
# import tensorflow as tf
# from keras.preprocessing import image
# import numpy as np

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'static/uploads'

# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# model = tf.keras.models.load_model('pneumonia_model.h5')

# def predict_image(img_path):
#     img = image.load_img(img_path, target_size=(150, 150))
#     img_tensor = image.img_to_array(img)
#     img_tensor = np.expand_dims(img_tensor, axis=0) / 255.0
#     prediction = model.predict(img_tensor)
#     return 'PNEUMONIA' if prediction[0][0] > 0.5 else 'NORMAL'

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     file = request.files['file']
#     if file:
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)
#         label = predict_image(filepath)
#         return render_template('result.html', image_file=filepath, label=label)
#     return redirect(url_for('index'))




# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model = tf.keras.models.load_model('pneumonia_model.h5')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0) / 255.0
    prediction = model.predict(img_tensor)
    return 'PNEUMONIA' if prediction[0][0] > 0.5 else 'NORMAL'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        label = predict_image(filepath)
        return render_template('result.html', image_file=filepath, label=label)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
