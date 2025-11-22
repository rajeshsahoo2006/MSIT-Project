import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model
import numpy as np
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class MNISTGAN:
    def __init__(self, batch_size: int = 128, random_dim: int = 100):
        self.batch_size = batch_size
        self.random_dim = random_dim
        self.x_train = self._load_data()
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        self.gan = self._build_gan()

    def _load_data(self):
        (x_train, _), _ = tf.keras.datasets.mnist.load_data()
        x_train = (x_train.astype("float32") - 127.5) / 127.5
        return x_train.reshape(60000, 784)

    def _build_generator(self):
        generator = Sequential([
            layers.Dense(256, input_dim=self.random_dim),
            layers.LeakyReLU(0.2),
            layers.Dense(512),
            layers.LeakyReLU(0.2),
            layers.Dense(784, activation="tanh"),
        ])
        return generator

    def _build_discriminator(self):
        discriminator = Sequential([
            layers.Dense(1024, input_dim=784),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.3),
            layers.Dense(512),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.3),
            layers.Dense(256),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
        ])
        discriminator.compile(loss="binary_crossentropy", optimizer="adam")
        return discriminator

    def _build_gan(self):
        self.discriminator.trainable = False
        z_input = layers.Input(shape=(self.random_dim,))
        generated = self.generator(z_input)
        validity = self.discriminator(generated)
        gan = Model(z_input, validity)
        gan.compile(loss="binary_crossentropy", optimizer="adam")
        return gan

    def train(self, epochs: int = 100):
        batch_count = self.x_train.shape[0] // self.batch_size

        for epoch in range(epochs):
            for _ in range(batch_count):
                noise = np.random.normal(0, 1, size=[self.batch_size, self.random_dim])
                generated_images = self.generator.predict(noise, verbose=0)

                idx = np.random.randint(0, self.x_train.shape[0], self.batch_size)
                real_images = self.x_train[idx]

                X = np.concatenate([real_images, generated_images])
                y = np.zeros(2 * self.batch_size)
                y[:self.batch_size] = 1

                self.discriminator.trainable = True
                self.discriminator.train_on_batch(X, y)

                noise = np.random.normal(0, 1, size=[self.batch_size, self.random_dim])
                y_gen = np.ones(self.batch_size)

                self.discriminator.trainable = False
                self.gan.train_on_batch(noise, y_gen)

            print(f"Epoch {epoch + 1}/{epochs}")

    def generate_images(self, sample_size: int = 100):
        noise = np.random.normal(0, 1, (sample_size, self.random_dim))
        gen_imgs = self.generator.predict(noise, verbose=0)
        return gen_imgs.reshape(sample_size, 28, 28)


if __name__ == "__main__":
    mnist_gan = MNISTGAN(batch_size=128, random_dim=100)
    mnist_gan.train(epochs=40)
    imgs = mnist_gan.generate_images(sample_size=25)  # Generate 25 images for 5x5 grid
    if HAS_MATPLOTLIB:
        # Create a 5x5 grid of images
        fig, axes = plt.subplots(5, 5, figsize=(10, 10))
        fig.suptitle('Generated MNIST Digits (5x5 Grid)', fontsize=16, fontweight='bold')
        
        for i in range(5):
            for j in range(5):
                idx = i * 5 + j
                axes[i, j].imshow(imgs[idx], cmap="gray")
                axes[i, j].axis("off")
        
        plt.tight_layout()
        plt.savefig('generated_digits_grid.png', dpi=150, bbox_inches='tight')
        print("Generated images saved to 'generated_digits_grid.png'")
        plt.show()
        
        # Also show a single image
        plt.figure(figsize=(6, 6))
        plt.imshow(imgs[0], cmap="gray")
        plt.title('Sample Generated Digit', fontsize=14)
        plt.axis("off")
        plt.show()
    else:
        print(f"Generated {len(imgs)} images. Shape: {imgs.shape}")
        print("Install matplotlib to visualize the generated images.")

