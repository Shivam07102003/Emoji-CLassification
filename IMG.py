
import os
import pandas as pd
from tensorflow.keras.utils import to_categorical
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import base64
from PIL import Image
import io
import imageio.v3 as iio
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

csv_file_path = 'Dataset/full_emoji.csv'
df = pd.read_csv(csv_file_path)\

def get_emoji_name(image_filename):
    image_id = os.path.splitext(image_filename)[0]
    
    if image_id.isdigit():
        image_id = int(image_id)
        emoji_row = df.iloc[image_id - 1]  
        return emoji_row['name']
    else:
        return "Emoji not found"
    
image_dir = 'Dataset/image/'
X = []
y = []

for image_file in os.listdir(image_dir):
    if image_file.endswith('.png'):
        image_path = os.path.join(image_dir, image_file)
        
        image = load_img(image_path, target_size=(64, 64)) 
        image = img_to_array(image)
        X.append(image)
        
        emoji_name = get_emoji_name(image_file)
        y.append(emoji_name)


X = np.array(X, dtype='float32') / 255.0  
y = np.array(y)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

y_one_hot = to_categorical(y_encoded)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X, y_one_hot, epochs=30, batch_size=32)

model.save('emoji_classifier_model.h5')

np.save('label_classes.npy', label_encoder.classes_)


def predict_emoji_name(image_path):

    image = load_img(image_path, target_size=(64, 64))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    
    model = tf.keras.models.load_model('emoji_classifier_model.h5')

    predictions = model.predict(image)
    predicted_label = np.argmax(predictions, axis=1)

    label_classes = np.load('label_classes.npy', allow_pickle=True)
    emoji_name = label_classes[predicted_label][0]
    
    return emoji_name


# Example prediction
image_path = "Dataset/image/1471.png"  # Replace with your image path
emoji_name = predict_emoji_name(image_path)
print(f"The predicted emoji name is: {emoji_name}")

