# Import Libraries
import shap
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a baseline RandomForestClassifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate the baseline model
y_pred = model.predict(X_test)
print("Baseline Model Accuracy:", accuracy_score(y_test, y_pred))

# Create a SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

# Plot feature importance
shap.summary_plot(shap_values, X_train, feature_names=feature_names)

# Custom function to wrap the scikit-learn model's predict_proba method
def sklearn_predict_proba(inputs):
    def predict(inputs):
        return model.predict_proba(inputs)
    return tf.py_function(predict, [inputs], tf.float32)

# Define the Keras model that uses the scikit-learn model
input_tensor = Input(shape=(4,))
output_tensor = Lambda(lambda x: sklearn_predict_proba(x))(input_tensor)
model_keras = Model(inputs=input_tensor, outputs=output_tensor)

# Generate Adversarial Examples using FGSM
def create_adversarial_pattern(input_data, input_label):
    with tf.GradientTape() as tape:
        tape.watch(input_data)
        prediction = model_keras(input_data)
        loss = tf.keras.losses.sparse_categorical_crossentropy(input_label, prediction)
    gradient = tape.gradient(loss, input_data)
    signed_grad = tf.sign(gradient)
    return signed_grad

# Generate adversarial examples
epsilon = 0.1
X_test_adv = X_test + epsilon * create_adversarial_pattern(tf.convert_to_tensor(X_test, dtype=tf.float32), tf.convert_to_tensor(y_test, dtype=tf.int64))
X_test_adv = np.clip(X_test_adv, 0, 1)

# Combine original and adversarial training data
X_train_adv = X_train + epsilon * create_adversarial_pattern(tf.convert_to_tensor(X_train, dtype=tf.float32), tf.convert_to_tensor(y_train, dtype=tf.int64))
X_train_adv = np.clip(X_train_adv, 0, 1)
X_train_combined = np.vstack((X_train, X_train_adv))
y_train_combined = np.hstack((y_train, y_train))

# Train a new RandomForestClassifier on the combined dataset
model_adv = RandomForestClassifier(n_estimators=100, random_state=42)
model_adv.fit(X_train_combined, y_train_combined)

# Evaluate and compare model performances
y_pred_adv = model_adv.predict(X_test)
print("Adversarially Trained Model Accuracy on Clean Data:", accuracy_score(y_test, y_pred_adv))

y_pred_adv_adv = model_adv.predict(X_test_adv)
print("Adversarially Trained Model Accuracy on Adversarial Data:", accuracy_score(y_test, y_pred_adv_adv))

y_pred_base_adv = model.predict(X_test_adv)
print("Baseline Model Accuracy on Adversarial Data:", accuracy_score(y_test, y_pred_base_adv))