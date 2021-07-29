import tensorflow as tf

print(tf.__version__)

mnist = tf.keras.datasets.mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

print(f"before normalise {training_images[3], training_labels[3]}")


def normalise_v1(x_img, y_img):
    training_images = x_img / 255.0
    test_images = y_img / 255.0
    return (training_images, test_images)


# training_images, test_images = normalise_v1(training_images, test_images)


def normalise_2(x_img, y_img):
    training_images = (x_img - x_img.min()) / (x_img.max() - x_img.min())
    test_images = (y_img - y_img.min()) / (y_img.max() - y_img.min())
    return (training_images, test_images)


# training_images, test_images = normalise_2(training_images, test_images)

print(f"after normalise {training_images[3], training_labels[3]}")
