#Author: Muhammad Azizi Bin Mohd Ariffin
#Email: mazizi@fskm.uitm.edu.my
#Description: GAN Attack Program for ISCX-VPN 2016 MLP

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import time

#load real network traffic from class 2
real_data = np.load("Client3class2data.npy")

print(real_data)

print(real_data.shape)

'''
# Generator Model
def make_generator_model(input_dim, output_dim):
    model = tf.keras.Sequential()
    model.add(layers.Dense(128, use_bias=False, input_shape=(input_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Dense(256))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Dense(output_dim, activation='tanh'))
    return model
'''
def make_generator_model(input_dim, output_dim):
    model = tf.keras.Sequential()
    model.add(layers.Dense(128, use_bias=False, input_shape=(input_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Dense(256))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Dense(512))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Dense(output_dim, activation='sigmoid'))
    return model
'''
# Discriminator Model
def make_discriminator_model(input_dim):
    model = tf.keras.Sequential()
    model.add(layers.Dense(128, input_shape=(input_dim,)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(64))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(1, activation='sigmoid'))
    return model
'''  
# Discriminator Model
def make_discriminator_model(input_dim):
    model = tf.keras.Sequential()
    model.add(layers.Dense(128, input_shape=(input_dim,)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(1, activation='sigmoid'))
    
    return model

# Loss and Optimizer
cross_entropy = tf.keras.losses.BinaryCrossentropy()

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
#generator_optimizer = tf.keras.optimizers.Adam(5e-5)

discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
#discriminator_optimizer = tf.keras.optimizers.Adam(5e-5)

# Training Step
@tf.function
def train_step(generator, discriminator, batch_size, noise_dim, real_data):
    noise = tf.random.normal([batch_size, noise_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_data = generator(noise, training=True)

        real_output = discriminator(real_data, training=True)
        fake_output = discriminator(generated_data, training=True)

        gen_loss = generator_loss(fake_output) #Discriminator Feedback to Generator
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

# Training Loop
def train(generator, discriminator, epochs, noise_dim, batch_size, real_data):
    for epoch in range(epochs):
        idx = np.random.randint(0, real_data.shape[0], batch_size)
        batch_data = tf.convert_to_tensor(real_data[idx])

        gen_loss, disc_loss = train_step(generator, discriminator, batch_size, noise_dim, batch_data)

        # Print the loss every 50 epochs
        if epoch % 50 == 0:
            print(f'Epoch {epoch}, Gen Loss: {gen_loss}, Disc Loss: {disc_loss}')

# Generate Synthetic Data
def generate_synthetic_data(generator, noise_dim, num_samples):
    noise = tf.random.normal([num_samples, noise_dim])
    generated_data = generator(noise, training=False)
    return generated_data.numpy()

# Initialize and train
noise_dim = 200
data_dim = real_data.shape[1]  # 740 in this case
epochs = 5000
batch_size = 32

generator = make_generator_model(noise_dim, data_dim)
discriminator = make_discriminator_model(data_dim)

'''
count = 0
for layer in generator.layers:
    print(layer.name, layer)
    print(generator.layers[count].weights)
    #print(generator.layers[count].bias.numpy())
    count += 1
    
print("break")

count = 0
for layer in discriminator.layers:
    print(layer.name, layer)
    print(discriminator.layers[count].weights)
    #print(discriminator.layers[count].bias.numpy())
    count += 1
'''

train(generator, discriminator, epochs, noise_dim, batch_size, real_data)

'''
print("After Training")
count = 0
for layer in generator.layers:
    print(layer.name, layer)
    print(generator.layers[count].weights)
    #print(generator.layers[count].bias.numpy())
    count += 1
    
print("break")

count = 0
for layer in discriminator.layers:
    print(layer.name, layer)
    print(discriminator.layers[count].weights)
    #print(discriminator.layers[count].bias.numpy())
    count += 1
'''

# Generate synthetic data samples
synthetic_data = generate_synthetic_data(generator, noise_dim, 20000)
print(synthetic_data.shape)
print(synthetic_data)
np.save('Client3generatedClass2x', synthetic_data)
print("This is sample of real data packet number 1")
print(real_data[0])
print("This is sample of synthetic data packet number 1")
print(synthetic_data[0])
print("This is sample of synthetic data packet number 2")
print(synthetic_data[1])