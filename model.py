import tensorflow as tf

class Model(object):
    def __init__(self, input_shape, hidden_layer_size, output_num):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=input_shape),
            tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
            tf.keras.layers.Dense(output_num)
        ])

    def compiled(self, learning_rate):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )
        return self

    def trained(self, training_dataset, validation_dataset, num_epochs):
        self.model.fit(
            training_dataset,
            epochs=num_epochs,
            validation_data=validation_dataset,
        )
        return self
