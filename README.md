# Multi-Class Classification of Obesity Levels Based on Eating Habits and Physical Condition

**Machine Learning (S2-25_DSECLZG565) — Assignment 2**
**BITS ID:** 2025DA04046

## Problem Statement

This project is designed to classify obesity levels using individuals' eating habits and physical conditions as attributes. The work has been completed using “Estimation of Obesity Levels Based On Eating Habits and Physical Condition”, which was collected from the UCI Machine Learning Repository. The mentioned dataset contains 16 features having demographic information like age, gender, height, weight; dietary habits like frequency of vegetable consumption, number of main meals, consumption of high-caloric food; physical activity patterns like frequency of physical activity, time using technology devices, mode of transportation. Here, the “NObeyesdad” attribute is targeted for classification, where it helps in classifying 7 categories ranging from Insufficient Weight to Obesity Type III. In this work, five supervised classifiers have been implemented: Logistic Regression, Decision Tree, K-Nearest Neighbors, Naive Bayes, and Random Forest. All of the models are trained and evaluated so that a comparative approach can be presented to identify the best-fitted model for the selected dataset, which predicts the obesity category from these behavioural and physical attributes.

## Description of Dataset

The work is completed using the “Estimation of Obesity Levels Based On Eating Habits and Physical Condition” dataset, which was collected from the UCI Machine Learning Repository. The dataset contains 2111 instances where 16 features are observed. The features are a mix of both numerical and categorical variables. The targeted feature is “NObeyesdad”, which contains 7 classes like Insufficient Weight, Normal Weight, Overweight Level I, Overweight Level II, Obesity Type I, Obesity Type II and Obesity Type III. 
While checking the dataset at the time of cleaning, we found no missing values and duplicate records in the dataset. While performing data processing, all of the numeric columns were verified and cast to the correct numeric types before classifying them. Additionally, while performing exploratory data analysis, we found the classes are distributed in a balanced way, and a strong correlation has been found between Height and Weight.

## GitHub Repository Link

https://github.com/arnabchatterjee1993/Classification-of-Obesity-Levels

## Models Used

In this work, five supervised classifiers have been implemented: Logistic Regression, Decision Tree, K-Nearest Neighbors, Naive Bayes, and Random Forest.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8900 | 0.9852 | 0.8872 | 0.8867 | 0.8836 | 0.8724 |
| Decision Tree | 0.9258 | 0.9553 | 0.9243 | 0.9229 | 0.9233 | 0.9135 |
| KNN | 0.8086 | 0.9429 | 0.7927 | 0.8011 | 0.7864 | 0.7797 |
| Naive Bayes | 0.5981 | 0.8867 | 0.6204 | 0.5932 | 0.5595 | 0.5482 |
| Random Forest (Ensemble) | 0.9545 | 0.9973 | 0.9548 | 0.9528 | 0.9531 | 0.9471 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Though the model has a good accuracy score, it also has a strong AUC (0.985). This indicates the model can make good probabilistic separation between classes. The performance struggles with the adjacent, borderline categories, which are not separable in this particular feature space. |
| Decision Tree | Observing the evaluation metrics, this model performs well and reported 0.917 accuracy and 0.904 MCC. This model naturally handles the categorical and numerical features without scaling them and is able to save non-linear splits between neighboring obesity classes. |
| KNN | Considering the evaluation metrics, it has achieved 0.806 accuracy for this particular dataset. This modelling approach is very much sensitive to the scaling and dimensionality of the 16 encoded features. This actually makes it noisier in calculating neighbours when it is compared with a tree-based model. |
| Naïve Bayes | Considering other implemented models, this model is the poorest performer and achieved 0.598 accuracy and 0.546 MCC after evaluation. We have observed earlier that features like Height and Weight are highly correlated, and this violated its core assumption of feature independence. It is unable to differentiate adjacent obesity classes here. |
| Random Forest (Ensemble) | Comparing the evaluation metrics of all implemented classification models, this model is the best-fitted approach for this dataset, and it resulted in the best score in terms of accuracy, AUC and MCC. As an ensemble of decision trees, it averages out overfitting issues and provides higher accuracy with nearly-perfect separation of classed whih is reflected in the AUC score. |

**Overall Winner:** All of the implemented models are trained and fitted with the dataset. In this work, it was found that Logistic Regression, Decision Tree, KNN, Naive Bayes and Random Forest achieve accuracy of 0.8676, 0.9173, 0.8061, 0.5981 and 0.9527, respectively. Comparing the performance of all of the implemented models, based on their evaluation matrices, we found that the Random Forest (Ensemble) is the best-fitted approach for this sort of work. It has been observed that Random Forest (Ensemble) is the best performer as it generalises this type of tabular data better than single estimators, which is expected from an ensemble method.

## Live Streamlit App Link

https://classification-of-obesity-levels.streamlit.app/

## Repository Structure

```
├── streamlit_app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── classify_obesity_models.ipynb
├── .gitignore
└── model/
    ├── classify_obesity_models.ipynb
    └── saved_models/
        ├── Logistic_Regression.pkl
        ├── Decision_Tree.pkl
        ├── KNN.pkl
        ├── Naive_Bayes.pkl
        ├── Random_Forest.pkl
        ├── scaler.pkl
        ├── encoders.pkl
        └── target_encoder.pkl
```

