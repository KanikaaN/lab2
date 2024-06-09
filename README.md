# lab2
Breast Cancer Wisconsin (Diagnostic) Data Set - Machine Learning Models

This repository contains a Jupyter notebook that demonstrates the process of building, training, and evaluating machine learning models using the Breast Cancer Wisconsin (Diagnostic) dataset.

## Dataset

The dataset used in this project is the Breast Cancer Wisconsin (Diagnostic) dataset, which is available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)). The dataset contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. The features describe characteristics of the cell nuclei present in the image.

## Files

- wdbc.data: The dataset file.
- Kanika_BreastCancerModel.ipynb: Jupyter notebook containing the code for loading the dataset, preprocessing, training the machine learning models, and evaluating their performance.
- Kanika_BreastCancerModel2.ipynb: (In the svm_model branch) Jupyter notebook containing the code for loading the dataset, preprocessing, training the machine learning models, and evaluating their performance.
- README.md: This file.

## Steps

### 1. Importing Libraries and Loading the Dataset

We start by importing the necessary libraries and loading the dataset from the wdbc.data file.

### 2. Data Preprocessing

In this step, we preprocess the data by:
- Dropping the 'ID' column as it is not useful for prediction.
- Encoding the 'Diagnosis' column where M (malignant) is mapped to 1 and B (benign) is mapped to 0.
- Checking for missing values and splitting the dataset into features (X) and target variable (y).

### 3. Splitting the Dataset

We split the dataset into training and testing sets using an 80-20 split.

### 4. Training Machine Learning Models

We train two different models:
- A RandomForestClassifier
- A Support Vector Machine (SVM) classifier

### 5. Evaluating the Models

We evaluate both models using accuracy, confusion matrix, and classification report.


## Model Performance

### Random Forest Classifier

- *Accuracy*: 0.96
- *Confusion Matrix*:
  
  [[70  1]
   [ 3 40]]
  
- *Classification Report*:
  
                precision    recall  f1-score   support

            0       0.96      0.99      0.97        71
            1       0.98      0.93      0.95        43

     accuracy                           0.96       114
    macro avg       0.97      0.96      0.96       114
 weighted avg       0.97      0.96      0.96       114
  

### SVM Classifier

- *Accuracy*: 0.96
- *Confusion Matrix*:
  
  [[70  1]
   [ 4 39]]
  
- *Classification Report*:
  
                precision    recall  f1-score   support

            0       0.95      0.99      0.97        71
            1       0.97      0.91      0.94        43

     accuracy                           0.96       114
    macro avg       0.96      0.95      0.96       114
 weighted avg       0.96      0.96      0.96       114
  

## Instructions

1. Ensure you have the dataset file wdbc.data in the same directory as your Jupyter notebook.
2. Run the breast_cancer_classification.ipynb notebook step by step to see the process of building and evaluating the models.
3. (Optional) The trained models are saved as breast_cancer_rf_model.pkl and breast_cancer_svm_model.pkl.

## Requirements

- Python 3.x
- Jupyter Notebook
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

