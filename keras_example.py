from dataset import Dataset
from model import Model

def main():
    dataset = Dataset('mnist')

    model = Model(
        input_shape=(28,28),
        hidden_layer_size=128,
        output_num=10).compiled(
            learning_rate=0.001).trained(
                training_dataset=dataset.train,
                validation_dataset=dataset.test,
                num_epochs=6)


if __name__ == "__main__":
    main()