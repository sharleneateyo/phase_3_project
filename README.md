# Predicting Water Well Functionality in Tanzania 🇹🇿
# Project Overview
This project aims to predict the functional status of water wells in Tanzania using a machine learning model. By leveraging a dataset from the Tanzania Ministry of Water, the goal is to shift from a costly reactive maintenance model to a proactive, data-driven strategy. The predictive model can help identify wells that are either non-functional or require repair, enabling more efficient resource allocation and maintenance planning

# Business Problem
Access to clean, safe water is a significant challenge in Tanzania, where many water points are in disrepair. The traditional approach of waiting for a well to completely fail before repairing it is inefficient. A predictive system can provide an early warning, allowing for timely repairs and ensuring consistent water access for communities. This project builds a classification model to categorize water wells into three states: functional, functional but needs repair, and non-functional.

# Dataset
The dataset contains records of water wells across Tanzania, with features describing their location, age, management, and technical specifications. The target variable, status_group, is a multi-class variable representing the condition of the well.

 ## Data Preprocessing & Feature Engineering
The raw data from two separate files was merged into a single DataFrame.

Missing values were handled, particularly in features like funder, installer, and permit.

Feature engineering was performed on fields like construction_year to treat a value of 0 as missing data.

some columns which were of no value to our model were dropped. such as date recorded, recorded by and columns which had similar values.

## Handling Class Imbalance
The target variable exhibited a significant class imbalance, which can negatively impact a model's ability to learn from the minority classes. To address this, the SMOTE (Synthetic Minority Over-sampling Technique) method was applied to the training data to balance the class distribution.

## Modeling & Hyperparameter Tuning
A Random Forest Classifier was chosen as the primary model. To find the optimal settings, RandomizedSearchCV was performed over a predefined parameter grid. The model's performance was evaluated using the recall, accuracy score and ROC curves to ensure balanced predictive power across all three classes.

## Key Findings
These project successfully built a machine learning model to predict the functional status of water wells in Tanzania. By using an XGBoost classifier and addressing class imbalance with SMOTE, the model achieved a solid overall accuracy of 79%. It performs very well in identifying functional and non-functional wells. However, the model has a critical weakness: it struggles to correctly identify wells that are functional but need repair. This is evidenced by the very low recall of only 48% for this class, meaning the model misses a large majority of the wells it was designed to help maintain proactively.

## Recommendations
 To improve the model's performance, particularly on the "needs repair" class, the following steps should be:

- Refine Class Imbalance Handling: The current approach with SMOTE may be introducing noisy or unrealistic data points that confuse the model. You could try an alternative strategy such as:
Class Weighting: Pass scale_pos_weight in XGBoost to directly give more importance to the minority class during training.

- Alternative Oversampling: Explore other techniques, which can be more effective than basic SMOTE.

- Conducting Deeper Feature Engineering and Selection:
by removing features with very low importance to simplify the model and potentially improve its performance.
- Broadening Hyperparameter Search: The n_estimators parameter was at the edge of the search range in your randomized search. This suggests that a larger number of trees could lead to better performance. Expand your search to include a wider range of values (e.g., [200, 300, 400, 500]) to see if the model's performance continues to improve.

- Explore Alternative Models: While XGBoost is a powerful model, others may be better suited for this specific problem. Consider training and evaluating a LightGBM Classifier, which is known for its speed and ability to handle large datasets and imbalanced classes effectively.

## Tools Used
Python: Data cleaning, preprocessing, exploratory analysis, and hypothesis testing (Pandas, Matplotlib, Seaborn, Scipy) and scikit-learn
