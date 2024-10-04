# Iris Dataset Classification with Adversarial Training and SHAP

## Objective
This project aims to classify the species of the Iris dataset using a machine learning pipeline with a Random Forest Classifier. The model is evaluated on clean and adversarial data, explained using SHAP for feature importance, and trained with adversarial examples to improve robustness.

## Techniques Used
1. **Baseline Model**: A `RandomForestClassifier` is trained on the Iris dataset, and its performance is evaluated on clean test data.
2. **SHAP Explainability**: SHAP (SHapley Additive exPlanations) is used to explain the model’s predictions and visualize the feature importance. This improves model transparency and interpretability.
3. **Adversarial Training**: The model is trained with adversarial examples generated using the Fast Gradient Sign Method (FGSM). This makes the model more robust against adversarial attacks.

## Adversarial Training
Adversarial training increases the robustness of machine learning models by including adversarial examples in the training process. Adversarial examples are slightly perturbed inputs that aim to fool the model. By training on both clean and adversarial data, the model learns to resist such attacks, improving its generalization and security.

## Dataset
- The dataset used is the Iris dataset, which can be loaded using the `sklearn.datasets.load_iris()` method.

## Files
- `improved_RF-iris_Shap.py`: Contains the code for training the RandomForest model, generating SHAP plots, creating adversarial examples using FGSM, and evaluating the model's performance on clean and adversarial test data.

## Requirements
To install the necessary dependencies, run:
```bash
pip install -r requirements.txt
