# Loan-Prediction-AnalyticsVidhya-Challenge

Competition Link:

**Feature analysis:**
1. Chisquare test to find feature dependence for categorical variables
2. Pearson correlation to find correlation between continuous variables
3. One way anova to find correlation between continuous and categorical variables

**Data is having many missing values so missing values imputation was performed in following ways:**

- For categorical variables
1. Treat NAs as separate categories
2. Replace by mode

- For continuous variables
1. Replace by mean of feature
2. Group average (Grouped by features found from one way ANOVA)

** Feature Transformation **
1. One hot encoding
2. Normalization
3. 2 way interactions
4. Manually crafted features

** Models **
1. Logistic Regression (Best model)
2. Random forest
3. Extra trees
4. Gradient boosting machine
5. Naive Bayes
6. K-nearest neighbours
7. Ensemble Stacking


Coding Structure and template reference: https://github.com/alno
