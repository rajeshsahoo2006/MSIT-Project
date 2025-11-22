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

    def train(self, epochs: int = 100, save_progression_at=None, fixed_noise=None):
        """
        Train the GAN model.
        
        Args:
            epochs: Number of epochs to train
            save_progression_at: List of epochs at which to save generated images
            fixed_noise: Fixed noise vector for consistent image generation across epochs
        """
        batch_count = self.x_train.shape[0] // self.batch_size
        progression_images = {}
        
        # Create fixed noise if not provided
        if fixed_noise is None:
            fixed_noise = np.random.normal(0, 1, (25, self.random_dim))
        
        if save_progression_at is None:
            save_progression_at = []

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

            current_epoch = epoch + 1
            print(f"Epoch {current_epoch}/{epochs}")
            
            # Save progression images at specified epochs
            if current_epoch in save_progression_at:
                gen_imgs = self.generator.predict(fixed_noise, verbose=0)
                gen_imgs = gen_imgs.reshape(25, 28, 28)
                progression_images[current_epoch] = gen_imgs
                print(f"  → Saved progression image at epoch {current_epoch}")
        
        return progression_images, fixed_noise

    def generate_images(self, sample_size: int = 100):
        noise = np.random.normal(0, 1, (sample_size, self.random_dim))
        gen_imgs = self.generator.predict(noise, verbose=0)
        return gen_imgs.reshape(sample_size, 28, 28)


if __name__ == "__main__":
    # Epochs to capture progression
    progression_epochs = [1, 20, 40, 140, 160, 200]
    
    mnist_gan = MNISTGAN(batch_size=128, random_dim=100)
    
    # Train with progression tracking
    print("Starting GAN training with progression tracking...")
    print(f"Will capture images at epochs: {progression_epochs}\n")
    
    progression_images, fixed_noise = mnist_gan.train(
        epochs=max(progression_epochs), 
        save_progression_at=progression_epochs,
        fixed_noise=None
    )
    
    if HAS_MATPLOTLIB:
        # Create progression visualization showing all epochs side by side
        num_epochs = len(progression_epochs)
        fig, axes = plt.subplots(5, num_epochs, figsize=(3*num_epochs, 15))
        fig.suptitle('GAN Progression: Generated Images at Different Epochs', 
                     fontsize=18, fontweight='bold', y=0.995)
        
        for col_idx, epoch in enumerate(progression_epochs):
            if epoch in progression_images:
                imgs = progression_images[epoch]
                
                # Set title for each epoch column
                axes[0, col_idx].set_title(f'Epoch {epoch}', fontsize=14, fontweight='bold', pad=10)
                
                # Display 5 sample images from each epoch (one per row)
                for row_idx in range(5):
                    img_idx = row_idx * 5  # Sample from different parts of the grid
                    axes[row_idx, col_idx].imshow(imgs[img_idx], cmap="gray")
                    axes[row_idx, col_idx].axis("off")
        
        plt.tight_layout()
        plt.savefig('gan_progression.png', dpi=200, bbox_inches='tight')
        print(f"\n✓ Progression visualization saved to 'gan_progression.png'")
        
        # Also create individual 5x5 grids for each epoch
        for epoch in progression_epochs:
            if epoch in progression_images:
                imgs = progression_images[epoch]
                fig, axes = plt.subplots(5, 5, figsize=(10, 10))
                fig.suptitle(f'Generated MNIST Digits at Epoch {epoch}', 
                           fontsize=16, fontweight='bold')
                
                for i in range(5):
                    for j in range(5):
                        idx = i * 5 + j
                        axes[i, j].imshow(imgs[idx], cmap="gray")
                        axes[i, j].axis("off")
                
                plt.tight_layout()
                filename = f'gan_epoch_{epoch}.png'
                plt.savefig(filename, dpi=150, bbox_inches='tight')
                print(f"✓ Saved {filename}")
                plt.close()
        
        # Show the main progression plot
        plt.show()
        
    else:
        print(f"\nProgression images captured at epochs: {list(progression_images.keys())}")
        print("Install matplotlib to visualize the generated images.")

