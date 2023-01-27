import tensorflow as tf
import tensorflow_datasets as tfds

class Dataset(object):
    def __init__(self, dataset):
        (self.train, self.test), self.info = tfds.load(
            'mnist',
            split=['train', 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
        )
        def normalize_img(image, label):
            """Normalizes images: `uint8` -> `float32`."""
            return tf.cast(image, tf.float32) / 255., label

        self.train = self.train.map(
            normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        self.train = self.train.cache()
        self.train = self.train.shuffle(self.info.splits['train'].num_examples)
        self.train = self.train.batch(128)
        self.train = self.train.prefetch(tf.data.AUTOTUNE)

        self.test = self.test.map(
            normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        self.test = self.test.batch(128)
        self.test = self.test.cache()
        self.test = self.test.prefetch(tf.data.AUTOTUNE)
