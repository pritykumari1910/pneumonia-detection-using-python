import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing import image

# Set path
base_dir = 'chest_xray'



# The shape of The Train data is: (5216, 2) 5216 total training images split between 2 classes
# The shape of The Test data is: (624, 2)
# The shape of The Validation data is: (16, 2)

# Preprocessing and data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,   # Normalize pixel values to [0, 1]
    zoom_range=0.2,   # Randomly zoom in/out
    horizontal_flip=True    # Randomly flip images horizontally
)

test_val_datagen = ImageDataGenerator(rescale=1./255)  # Only rescale for validation and test data

train_generator = train_datagen.flow_from_directory(
    os.path.join(base_dir, 'train'),  # Directory with training data
    target_size=(150, 150),   # Resize images to 150x150
    batch_size=32,       # Number of images to process in a batch
    class_mode='binary'    # Binary classification (pneumonia vs normal)
)

val_generator = test_val_datagen.flow_from_directory(
    os.path.join(base_dir, 'val'),  # Directory with validation data
    target_size=(150, 150),  # Resize images to 150x150
    batch_size=32,    # Number of images to process in a batch
    class_mode='binary'   # Binary classification (pneumonia vs normal)
)

test_generator = test_val_datagen.flow_from_directory(
    os.path.join(base_dir, 'test'),  # Directory with test data
    target_size=(150, 150),  # Resize images to 150x150
    batch_size=32,   # Number of images to process in a batch
    class_mode='binary', # Binary classification (pneumonia vs normal)
    shuffle=False # Do not shuffle test data for evaluation
)

# CNN Model
# MaxPooling after each convolution to reduce spatial dimensions.

# Flatten: converts 2D output to 1D.

# Dropout: randomly deactivates 50% neurons (prevents overfitting).

# Dense: fully connected layers.

# sigmoid for binary classification output (0–1).


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)), # input shape for images
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'), 
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# adam: optimizer for training.

# binary_crossentropy: loss function for binary classification.

# accuracy: performance metric.

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training

# Trains the model for 10 epochs using training and validation data.

# Stores training history in history.

history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

# Evaluation
loss, accuracy = model.evaluate(test_generator) # Evaluate the model on the test set
print(f"Test Accuracy: {accuracy*100:.2f}%") # Print test accuracy

# Save model
model.save('pneumonia_model.h5') # Save the trained model

# Plot accuracy & loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Model Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.legend()

plt.show()

# Predict random image from test set
import random

files = test_generator.filepaths
random_file = random.choice(files)

img = image.load_img(random_file, target_size=(150, 150))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)[0][0]
label = "PNEUMONIA" if prediction > 0.5 else "NORMAL"

plt.imshow(img)
plt.title(f"Prediction: {label}")
plt.axis('off')
plt.show()
