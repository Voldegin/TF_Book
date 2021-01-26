import urllib.request
import zipfile
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip"
train_filename = r"F:\VOLD\ML\TF Book\Chapter 2\horse_human\horse-or-human.zip"
train_dir = r"F:\VOLD\ML\TF Book\Chapter 2\horse_human\training"

validation_url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/validation-horse-or-human.zip"
validation_filename = r"F:\VOLD\ML\TF Book\Chapter 2\horse_human\validation-horse-or-human.zip"
validation_dir = r"F:\VOLD\ML\TF Book\Chapter 2\horse_human\validation"


def create_generator(url, filename, directory):
    urllib.request.urlretrieve(url, filename)

    zip_ref = zipfile.ZipFile(filename, 'r')
    zip_ref.extractall(directory)
    zip_ref.close()

    generator_func = ImageDataGenerator(rescale=1 / 255)
    generator = generator_func.flow_from_directory(directory, target_size=(300, 300), class_mode='binary')
    return generator


def define_model():
    train_model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    train_model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
        loss=tf.keras.losses.binary_crossentropy,
        metrics=['accuracy']
    )
    return train_model


# validation_generator = create_generator(validation_url, validation_filename, validation_dir)
# train_generator = create_generator(train_url, train_filename, train_dir)

# model = define_model()
# model.fit(train_generator, epochs=15, validation_data=validation_generator)

model_path = r"F:\VOLD\ML\TF Book\Chapter 2\horse_human"
# model.save(model_path)

# Load Model

model = tf.keras.models.load_model(model_path)

test_folder = r"F:\VOLD\ML\TF Book\Chapter 2\horse_human\test"

for test_image in os.listdir(test_folder):
    img = image.load_img(os.path.join(test_folder, test_image), target_size=(300, 300))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    image_tensor = np.vstack([x])
    classes = model.predict(image_tensor)
    # print(classes)
    # print(classes[0])
    if classes[0] > 0.5:
        print(test_image, "----human")
    else:
        print(test_image, "----horse")
