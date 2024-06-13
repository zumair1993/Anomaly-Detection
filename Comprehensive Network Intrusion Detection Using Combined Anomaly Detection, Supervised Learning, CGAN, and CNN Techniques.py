# Author: Umair Zia

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras import layers

# Define popular intrusion detection datasets
SUPPORTED_DATASETS = {
    'CICIDS2017': 'https://www.unb.ca/cicids/datasets/cicids2017.html',
    'UNSW-NB15': 'https://research.unb.ca/cicids/datasets/cicids2017.html'
}

def load_data(dataset_name):
    """
    Loads network traffic data from a CSV file.

    Args:
        dataset_name (str): Name of the supported dataset.

    Returns:
        pandas.DataFrame: Dataframe containing network traffic features.
    """
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    # Load the dataset (replace the path with actual data location)
    data = pd.read_csv('network_traffic.csv')
    return data

def preprocess_data(data):
    """
    Preprocesses network traffic data.

    Args:
        data (pandas.DataFrame): Dataframe containing network traffic features.

    Returns:
        pandas.DataFrame: Preprocessed dataframe.
    """
    # Separate features and labels
    X = data.drop('Label', axis=1)
    y = data['Label']
    # Normalize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns), y

def train_anomaly_detection(data):
    """
    Trains anomaly detection models.

    Args:
        data (pandas.DataFrame): Preprocessed network traffic data.

    Returns:
        dict: Dictionary containing trained anomaly detection models.
    """
    X_train, _ = train_test_split(data, test_size=0.2, random_state=42)

    # Train One-Class SVM model
    svm_model = OneClassSVM(nu=0.1)
    svm_model.fit(X_train)

    # Train Isolation Forest model
    if_model = IsolationForest(contamination=0.1, random_state=42)
    if_model.fit(X_train)

    # Train Local Outlier Factor model
    lof_model = LocalOutlierFactor(novelty=True, contamination=0.1)
    lof_model.fit(X_train)

    return {'OneClassSVM': svm_model, 'IsolationForest': if_model, 'LocalOutlierFactor': lof_model}

def train_supervised_models(X_train, y_train):
    """
    Trains supervised learning models.

    Args:
        X_train (pandas.DataFrame): Training features.
        y_train (pandas.Series): Training labels.

    Returns:
        dict: Dictionary containing trained supervised learning models.
    """
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)

    gb_model = GradientBoostingClassifier(random_state=42)
    gb_model.fit(X_train, y_train)

    return {'RandomForestClassifier': rf_model, 'GradientBoostingClassifier': gb_model}

def build_generator(latent_dim, num_features):
    """
    Builds the generator for CGAN.

    Args:
        latent_dim (int): Dimension of the latent space.
        num_features (int): Number of features in the dataset.

    Returns:
        tensorflow.keras.Sequential: Generator model.
    """
    model = tf.keras.Sequential([
        layers.Dense(128, activation='relu', input_dim=latent_dim),
        layers.Dense(256, activation='relu'),
        layers.Dense(num_features, activation='sigmoid')
    ])
    return model

def build_discriminator(num_features):
    """
    Builds the discriminator for CGAN.

    Args:
        num_features (int): Number of features in the dataset.

    Returns:
        tensorflow.keras.Sequential: Discriminator model.
    """
    model = tf.keras.Sequential([
        layers.Dense(256, activation='relu', input_dim=num_features),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

def train_cgan(generator, discriminator, data, epochs=10000, batch_size=64):
    """
    Trains the Conditional Generative Adversarial Network (CGAN).

    Args:
        generator (tensorflow.keras.Sequential): Generator model.
        discriminator (tensorflow.keras.Sequential): Discriminator model.
        data (pandas.DataFrame): Preprocessed network traffic data.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.

    Returns:
        tuple: Trained generator and discriminator models.
    """
    latent_dim = 100
    for epoch in range(epochs):
        # Train discriminator
        real_samples = data.sample(batch_size)
        noise = tf.random.normal([batch_size, latent_dim])
        fake_samples = generator(noise)
        with tf.GradientTape() as tape:
            real_output = discriminator(real_samples)
            fake_output = discriminator(fake_samples)
            d_loss = -tf.reduce_mean(real_output) + tf.reduce_mean(fake_output)
        grads = tape.gradient(d_loss, discriminator.trainable_variables)
        discriminator.optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

        # Train generator
        noise = tf.random.normal([batch_size, latent_dim])
        with tf.GradientTape() as tape:
            fake_samples = generator(noise)
            fake_output = discriminator(fake_samples)
            g_loss = -tf.reduce_mean(fake_output)
        grads = tape.gradient(g_loss, generator.trainable_variables)
        generator.optimizer.apply_gradients(zip(grads, generator.trainable_variables))
    return generator, discriminator

def build_cnn_model(input_shape):
    """
    Builds the Convolutional Neural Network (CNN) model.

    Args:
        input_shape (tuple): Input shape for the CNN model.

    Returns
