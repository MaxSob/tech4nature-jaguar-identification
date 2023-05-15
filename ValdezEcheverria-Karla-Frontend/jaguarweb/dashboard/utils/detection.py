import tensorflow as tf


def loadImage(file):
    
    img = tf.image.decode_jpeg(file, channels=3)
    img = tf.image.resize(img, (64, 64))
    img = tf.expand_dims(img, 0)

    return img