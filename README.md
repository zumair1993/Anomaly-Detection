# Comprehensive Network Intrusion Detection Using Combined Anomaly Detection, Supervised Learning, CGAN, and CNN Techniques
# Author Umair Zia
**Project Description:**
Network intrusion detection systems (NIDS) are essential tools for identifying and mitigating malicious activities in a network. This project involves loading popular network traffic datasets, preprocessing the data, training multiple anomaly detection and supervised learning models, and visualizing the results to effectively detect network intrusions. Additionally, we will incorporate Conditional Generative Adversarial Networks (CGAN) for data augmentation and Convolutional Neural Networks (CNN) using TensorFlow for enhanced detection capabilities.
Step-by-Step Detailed Description:
**1. Loading Data:**
Objective: Load network traffic data from popular datasets such as CICIDS2017 and UNSW-NB15.
Implementation: We use the pandas library to read data from a CSV file. The dataset contains various features of network traffic and a label indicating whether the traffic is normal or an attack.

**CODE:**
import pandas as pd

SUPPORTED_DATASETS = {
    'CICIDS2017': 'https://www.unb.ca/cicids/datasets/cicids2017.html',
    'UNSW-NB15': 'https://research.unb.ca/cicids/datasets/cicids2017.html'
}

def load_data(dataset_name):
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    # Load the dataset (replace the path with actual data location)
    data = pd.read_csv('network_traffic.csv')
    return data


**2. Preprocessing Data:**
Objective: Prepare the network traffic data for model training by separating features and labels, and normalizing the features.
Implementation: We separate the dataset into features (X) and labels (y). The features are then normalized using StandardScaler to ensure all features contribute equally to the models.

**CODE:**

from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    # Separate features and labels
    X = data.drop('Label', axis=1)
    y = data['Label']
    # Normalize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns), y


**3. Training Anomaly Detection Models:**

**Objective:**
Train unsupervised anomaly detection models to identify outliers in the network traffic data.
Implementation: We employ three different models: One-Class SVM, Isolation Forest, and Local Outlier Factor (LOF).
One-Class SVM: Fits a model to the training data and identifies outliers as anomalies.
Isolation Forest: Uses random feature selection and data splits to isolate anomalies.
Local Outlier Factor (LOF): Measures local density deviation of a data point compared to its neighbors.

**CODE:**

from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split

def train_one_class_svm(data):
    X_train, _ = train_test_split(data, test_size=0.2, random_state=42)
    model = OneClassSVM(nu=0.1)
    model.fit(X_train)
    return model

def train_isolation_forest(data):
    X_train, _ = train_test_split(data, test_size=0.2, random_state=42)
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X_train)
    return model

def train_lof(data):
    X_train, _ = train_test_split(data, test_size=0.2, random_state=42)
    model = LocalOutlierFactor(novelty=True, contamination=0.1)
    model.fit(X_train)
    return model

**4. Training Supervised Prediction Models:**

**Objective:** 
Train supervised learning models on labeled network traffic data to classify traffic as normal or anomalous.
Implementation: We use two models: Random Forest Classifier and Gradient Boosting Classifier.
Random Forest Classifier: An ensemble method using multiple decision trees to improve classification accuracy.
Gradient Boosting Classifier: Builds an ensemble of trees incrementally to minimize a loss function.

**CODE:**

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def train_prediction_models(X_train, y_train):
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    gb_model = GradientBoostingClassifier(random_state=42)
    gb_model.fit(X_train, y_train)
    return rf_model, gb_model

**5. Training Conditional Generative Adversarial Network (CGAN) for Data Augmentation:**

**Objective:**

Use CGAN to generate synthetic network traffic data for augmenting the dataset and improving model robustness.
Implementation: We use TensorFlow to build and train the CGAN. The CGAN consists of a generator and a discriminator. The generator creates synthetic data, and the discriminator evaluates its authenticity.

**CODE:**

import tensorflow as tf
from tensorflow.keras import layers

def build_generator(latent_dim, num_features):
    model = tf.keras.Sequential([
        layers.Dense(128, activation='relu', input_dim=latent_dim),
        layers.Dense(256, activation='relu'),
        layers.Dense(num_features, activation='sigmoid')
    ])
    return model

def build_discriminator(num_features):
    model = tf.keras.Sequential([
        layers.Dense(256, activation='relu', input_dim=num_features),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

def train_cgan(generator, discriminator, data, epochs=10000, batch_size=64):
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

**6. Training Convolutional Neural Network (CNN) for Intrusion Detection:**

**Objective:**

Use CNN for enhanced feature extraction and classification of network traffic data.
Implementation: We use TensorFlow to build and train the CNN. The CNN model consists of convolutional layers for feature extraction and dense layers for classification.

**CODE:**

def build_cnn_model(input_shape):
    model = tf.keras.Sequential([
        layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(64, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_cnn_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))
    return history

**7. Detecting Anomalies:**

**Objective:**

Identify anomalies in new network traffic data using the trained anomaly detection models.
Implementation: The models predict whether data points are normal or anomalous. Anomalous points are those predicted as outliers.

**CODE:**

def detect_anomalies(model, data):
    predictions = model.predict(data)
    anomaly_indices = data.index[predictions == -1].tolist()
    return anomaly_indices

**8. Evaluating Supervised Models:****

**Objective:**

Evaluate the performance of supervised learning models using metrics such as precision, recall, and F1-score.
Implementation: We split the data into training and testing sets, train the models, and generate classification reports.

**CODE:**

from sklearn.metrics import classification_report

def evaluate_models(models, X_test, y_test):
    for model in models:
        predictions = model.predict(X_test)
        print(f"{model.__class__.__name__} Classification Report:")
        print(classification_report(y_test, predictions))

**9. Visualizing Data and Results:**

**Objective:**
Visualize the distribution of features and the performance of supervised models.
Implementation: We use histograms, KDE plots, and ROC curves to visualize feature distributions and model performance.
