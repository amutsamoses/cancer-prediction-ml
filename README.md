# cancer-prediction-ml

# Breast Cancer Prediction Using Machine Learning

## Project Overview
This project is focused on predicting the diagnosis of breast cancer (malignant or benign) using machine learning models. We utilize various techniques for data preprocessing, exploration, visualization, model training, hyperparameter tuning, and deployment to create a machine learning solution for breast cancer prediction.

The dataset used in this project includes attributes related to cell nuclei, such as radius, texture, perimeter, area, smoothness, and others. These features help in distinguishing between malignant and benign tumor cases.

## Steps Covered

### 1. Gathering Data
The dataset used is from the UCI Machine Learning Repository, which contains various features related to cell nuclei. The main goal is to classify the tumors into two categories: malignant (M) and benign (B).

### 2. Exploratory Data Analysis (EDA)
- Load and clean the data.
- Handle missing values, if any.
- Examine data types and understand basic statistics.

### 3. Data Visualizations
Data visualizations help in understanding the relationship between different features and the target variable (Diagnosis). Key visualizations include:
- Histograms for feature distributions.
- Heatmaps for correlations between features.
- Pair plots for feature comparisons.

### 4. Model Implementation
- We used RandomForestClassifier as our primary machine learning model. 
- The model is trained and validated using a 10-fold cross-validation technique to ensure robust results.

### 5. ML Model Selection and Prediction
The RandomForestClassifier is selected due to its high accuracy and ability to handle complex data with multiple features. Predictions are made on the test set, and model performance is evaluated using accuracy and confusion matrix.

### 6. Hyperparameter Tuning
Using GridSearchCV, we tune the model's hyperparameters, such as `max_depth`, `n_estimators`, and others, to optimize model performance.

### 7. Deploy Model
- The trained model is exported using **pickle** for easy deployment.
- A Flask web API is built to serve the model and predict the diagnosis based on user input.

## Dataset Description
The dataset consists of the following attributes:

- ID number: An identifier for the sample.
- Diagnosis: The diagnosis result (M = malignant, B = benign).
- Ten real-valued features** (computed for each cell nucleus):
  - radius: Mean of distances from center to points on the perimeter.
  - texture: Standard deviation of gray-scale values.
  - perimeter: Perimeter of the cell.
  - area: Area of the cell.
  - smoothness: Local variation in radius lengths.
  - compactness: Perimeter^2 / area - 1.0.
  - concavity: Severity of concave portions of the contour.
  - concave points: Number of concave portions of the contour..
  - symmetry: Symmetry of the cell.
  - fractal dimension: Coastline approximation - 1.

## Getting Started

### Requirements

- Python 3.x
- Libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - Flask
  - pickle

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/amutsamoses/cancer-prediction-ml.git


### Implemented Breast Cancer Prediction using Machine Learning

- Completed exploratory data analysis and data visualizations.
- Trained and evaluated RandomForestClassifier model.
- Hyperparameter tuning with GridSearchCV for optimal performance.
- Saved the trained model using Pickle for deployment.
