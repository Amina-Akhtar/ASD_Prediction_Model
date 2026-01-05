# ASD Prediction Model

Autistic Spectrum Disorder (ASD) is a neurodevelopmental  condition associated with significant healthcare costs. This project uses machine learning techniques for the early diagnosis of ASD in toddlers using a dataset from Kaggle.

For information about dataset:
[Dataset Description](https://www.kaggle.com/datasets/fabdelja/autism-screening-for-toddlers)


## Methodology

- Conducted Exploratory Data Analysis-EDA to understand features distribution, class imbalance, and underlying patterns in the dataset.
- Performed Data visualization and Normalization to analyze feature relationships and ensure consistent scaling for model training.
- Applied Feature Selection Techniques to retain the most relevant features and reduce dimensionality.
- Evaluated multiple machine learning models and optimized hyperparameters using GridSearchCV with k-fold cross validation.
- Implemented a stacked ensemble using three base models and one meta-model to improve predictive performance.
- Assessed model performance using Accuracy, Recall, Precision, F1-Score, and ROC-AUC metrics.
- Saved the trained model, scalers, and encoders for reuse during inference.
- FastAPI to provide real-time predictions.

## Technologies Used:

Python, TensorFlow, Keras, Scikit-learn, FastAPI