import urllib.request
import zipfile
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3

train_url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps.zip"
train_filename = r"F:\VOLD\ML\TF Book\Chapter 2\rock_paper_scissors\rps.zip"
train_dir = r"F:\VOLD\ML\TF Book\Chapter 2\rock_paper_scissors\training"

validation_url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps-validation.zip"
validation_filename = r"F:\VOLD\ML\TF Book\Chapter 2\horse_human\validation-rps.zip"
validation_dir = r"F:\VOLD\ML\TF Book\Chapter 2\rock_paper_scissors\validation"


def create_generator(url, filename, directory):
    # urllib.request.urlretrieve(url, filename)
    #
    # zip_ref = zipfile.ZipFile(filename, 'r')
    # zip_ref.extractall(directory)
    # zip_ref.close()

    generator_func = ImageDataGenerator(rescale=1 / 255, rotation_range=50, width_shift_range=0.3,
                                        height_shift_range=0.3, shear_range=0.3, zoom_range=0.3,
                                        horizontal_flip=True, fill_mode='nearest'
                                        )
    generator = generator_func.flow_from_directory(directory, target_size=(150, 150), class_mode='categorical')
    return generator


def define_model():
    train_model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    train_model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=['accuracy']
    )
    return train_model


validation_generator = create_generator(validation_url, validation_filename, validation_dir)
train_generator = create_generator(train_url, train_filename, train_dir)

model = define_model()
model.fit(train_generator, epochs=25, validation_data=validation_generator)

model_path = r"F:\VOLD\ML\TF Book\Chapter 2\rock_paper_scissors"
model.save(model_path)

# Load Model

model = tf.keras.models.load_model(model_path)

test_folder = r"F:\VOLD\ML\TF Book\Chapter 2\rock_paper_scissors\test"

for test_image in os.listdir(test_folder):
    img = image.load_img(os.path.join(test_folder, test_image), target_size=(150, 150))
    img_x = image.img_to_array(img)
    img_x = np.expand_dims(img_x, axis=0)

    image_tensor = np.vstack([img_x])
    classes = model.predict(image_tensor)
    print(test_image, "----",classes)
