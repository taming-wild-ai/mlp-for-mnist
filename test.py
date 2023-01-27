import unittest

class TestMNISTTraining(unittest.TestCase):
    def test_training(self):
        from dataset import Dataset
        from model import Model
        import tensorflow as tf

        tf.keras.utils.set_random_seed(1234)

        dataset = Dataset('mnist')

        model = Model(
            input_shape=(28,28),
            hidden_layer_size=128,
            output_num=10).compiled(
                learning_rate=0.001).trained(
                    training_dataset=dataset.train,
                    validation_dataset=dataset.test,
                    num_epochs=1)

        _, accuracy = model.model.evaluate(dataset.test)
        self.assertAlmostEqual(accuracy, 0.9424, 2)

if __name__ == '__main__':
    unittest.main()
