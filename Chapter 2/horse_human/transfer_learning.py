import urllib.request
import zipfile
import shutil
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3

train_url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip"
train_filename = r"F:\VOLD\ML\TF Book\Chapter 2\horse_human\horse-or-human.zip"
train_dir = r"F:\VOLD\ML\TF Book\Chapter 2\horse_human\training"

validation_url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/validation-horse-or-human.zip"
validation_filename = r"F:\VOLD\ML\TF Book\Chapter 2\horse_human\validation-horse-or-human.zip"
validation_dir = r"F:\VOLD\ML\TF Book\Chapter 2\horse_human\validation"


def create_generator(url, filename, directory):
    # urllib.request.urlretrieve(url, filename)

    # zip_ref = zipfile.ZipFile(filename, 'r')
    # zip_ref.extractall(directory)
    # zip_ref.close()

    generator_func = ImageDataGenerator(rescale=1 / 255, rotation_range=40, width_shift_range=0.2,
                                        height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                                        horizontal_flip=True, fill_mode='nearest'
                                        )
    generator = generator_func.flow_from_directory(directory, target_size=(150, 150), class_mode='binary')
    return generator


def define_model():
    weights_url = "https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
    weights_file = r"F:\VOLD\ML\TF Book\Chapter 2\horse_human\transfer_learning\inception_v3.h5"
    urllib.request.urlretrieve(weights_url, weights_file)

    pre_trained_model = InceptionV3(input_shape=(150, 150, 3), include_top=False, weights=None)
    pre_trained_model.load_weights(weights_file)
    pre_trained_model.summary()

    for layer in pre_trained_model.layers:
        layer.trainable = False

    last_layer = pre_trained_model.get_layer('mixed7')
    print("Last Layer Output Shape:", last_layer.output_shape)
    last_output = last_layer.output

    x = tf.keras.layers.Flatten()(last_output)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    # x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    train_model = tf.keras.Model(pre_trained_model.input, x)

    train_model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
        loss=tf.keras.losses.binary_crossentropy,
        metrics=['accuracy']
    )
    return train_model


validation_generator = create_generator(validation_url, validation_filename, validation_dir)
train_generator = create_generator(train_url, train_filename, train_dir)

model = define_model()
model.fit(train_generator, epochs=40, validation_data=validation_generator)

model_path = r"F:\VOLD\ML\TF Book\Chapter 2\horse_human\transfer_learning"
shutil.rmtree(model_path)
os.makedirs(model_path)
model.save(model_path)

# Load Model

model = tf.keras.models.load_model(model_path)

test_folder = r"F:\VOLD\ML\TF Book\Chapter 2\horse_human\test"

for test_image in os.listdir(test_folder):
    img = image.load_img(os.path.join(test_folder, test_image), target_size=(150, 150))
    img_x = image.img_to_array(img)
    img_x = np.expand_dims(img_x, axis=0)

    image_tensor = np.vstack([img_x])
    classes = model.predict(image_tensor)
    # print(classes)
    # print(classes[0])
    if classes[0] > 0.5:
        print(test_image, "----human")
    else:
        print(test_image, "----horse")
