import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses

class Autoencoder(Model):
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim   

    self.encoder = tf.keras.Sequential([
      layers.Dense(384, activation='relu'),

      
      layers.Dense(192, activation='relu'),

      
      layers.Dense(96, activation='relu'),

      
      layers.Dense(48, activation='relu'),

      
      layers.Dense(latent_dim, activation='relu'),
      
    ])

    self.decoder = tf.keras.Sequential([
      layers.Dense(96, activation='relu'),

      
      layers.Dense(192, activation='relu'),

      
      layers.Dense(384, activation='relu'),

      
      layers.Dense(256*3)
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
