#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 20:23:28 2024

@author: tianyongwang
"""
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu
import pandas as pd
import random

#random number generator of my N number
n_number = 13173065
np.random.seed(n_number)
random.seed(n_number)


# Load numerical dataset
# Load the file with different options
data_num = pd.read_csv(
    "/Users/tianyongwang/Desktop/rmpCapstoneNum (2).csv",
   encoding="latin-1",
    header=None
)

data_qual = pd.read_csv(
    "/Users/tianyongwang/Desktop/rmpCapstoneQual (2).csv",
   encoding="latin-1",
    header=None
)

#data_num.head()

#changing column names 
data_num.columns = ["avg_rating", "avg_difficulty", "num_of_ratings","pepper","take_again_proportion","num_online","male","female"] 
data_qual.columns = ["Major/Field", "University","US State"] 
data_num.head()
data_qual.head()

#Question 1: Whether there is evidence of a pro-male gender bias in this dataset. 
#using mann whitney u test
from scipy.stats import mannwhitneyu
import seaborn as sns
import matplotlib.pyplot as plt

male_ratings = data_num[data_num["male"]==1]["avg_rating"]
female_ratings = data_num[data_num["female"]==1]["avg_rating"]

#mann-whitney u test
stat, p = mannwhitneyu(male_ratings, female_ratings, alternative="two-sided")

print("Question 1")
print(f"U statistic: {stat}")
print(f"p-value:{p}") 

sns.boxplot(
    data=data_num,
    x=data_num["male"].replace({1: "Male", 0: "Female"}), 
    y="avg_rating",
    hue=data_num["male"].replace({1: "Male", 0: "Female"}),  # Set 'hue' to gender
    palette={"Male": "skyblue", "Female": "lightpink"}  # Define custom colors
)
plt.title("Comparison of Average Ratings by Gender")
plt.xlabel("Gender")
plt.ylabel("Average Rating")

# Show the plot
plt.show()

#p-value : 2.443998308992039e-05
#> 0.0.5. Not siginficant

#Question 2: 
#Is there an effect of experince on the quality of teaching? 
#you can operationlize quality with the rating and use number of ratings as an imperfect
from scipy.stats import pearsonr
data_cleaned = data_num.dropna(subset=["num_of_ratings", "avg_rating"])
quality_of_teaching = data_cleaned["avg_rating"]
experience_of_teaching = data_cleaned["num_of_ratings"]
corr, pval = pearsonr(experience_of_teaching, quality_of_teaching) 
print("Question 2")
print("Pearson correlation coefficient ", corr)
print("p-value: ", pval)
print(" ")
import matplotlib.pyplot as plt
import seaborn as sns
 
sns.scatterplot(x=experience_of_teaching, y=quality_of_teaching)
plt.title("Experience vs Quality of Teaching (Average Rating)")
plt.xlabel("Number of Ratings (Experience)")
plt.ylabel("Average Rating (Quality of Teaching)")
plt.show()


#Question 3 
#whats the relationship between average rating and average difficulty? 
import scipy.stats as stats
data_cleaned1 = data_num.dropna(subset = ["avg_rating", "avg_difficulty"])
avg_rating =data_cleaned1["avg_rating"]
avg_diff = data_cleaned1["avg_difficulty"]
correlation, p_value = stats.pearsonr(avg_diff, avg_rating)
print("Question 3")
print("correlation: ", correlation)
print("p_value", p_value)
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(8, 6))
sns.scatterplot(x=avg_diff, y=avg_rating)
plt.title('Average Rating vs. Average Difficulty')
plt.xlabel('Average Difficulty')
plt.ylabel('Average Rating')
plt.show()
print("")


#Question 4 
#Do the professors who teach a lot of classes in the online modality 
#receive higher or lower ratings than those who don't? 
data_num['online_proportion'] = data_num['num_online'] / data_num['num_of_ratings']

data_num = data_num.replace([np.inf, -np.inf], np.nan).dropna(subset=["avg_rating", "online_proportion"])

threshold = data_num['online_proportion'].median()

high_online_group = data_num[data_num['online_proportion'] > threshold]['avg_rating']
low_online_group = data_num[data_num['online_proportion'] <= threshold]['avg_rating']

t_stat, p_value = ttest_ind(high_online_group, low_online_group, equal_var=False)

print("Question 4")
print(f"High Online Group Mean Rating: {high_online_group.mean():.2f}")
print(f"Low Online Group Mean Rating: {low_online_group.mean():.2f}")
print(f"T-Test Statistic: {t_stat:.4f}, P-Value: {p_value:.4f}")

data_num['online_group'] = data_num['online_proportion'].apply(lambda x: 'High Online' if x > threshold else 'Low Online')

sns.boxplot(x='online_group', y='avg_rating', data=data_num)
plt.title("Comparison of Average Ratings: High vs. Low Online Teaching")
plt.xlabel("Online Teaching Group")
plt.ylabel("Average Rating")
plt.show()


#Question 5 
#Whats the relationship between the average rating and the proportion of people who would take the class again? 
data_cleaned2 = data_num.dropna(subset = ["avg_rating", "take_again_proportion"])
avg_rating1 = data_cleaned2["avg_rating"]
proportion = data_cleaned2["take_again_proportion"]
correlation1, p_value1 = stats.pearsonr(avg_rating1, proportion)
print("Question 5")
print("correlation: ", correlation)
print("p_value", p_value)
sns.scatterplot(x='avg_rating', y='take_again_proportion', data=data_num)
plt.title("Relationship between Average Rating and Take Again Proportion")
plt.xlabel("Average Rating")
plt.ylabel("Proportion of People Willing to Take Class Again")
plt.show()
print("")

#Question 6
#Do professors who are hot receive higher ratings than those who are not? 
#from chat: the U-statistics measures the diff between ranks of 
#the values in two groups. Specifically, compares the distribution 
#of values between the two groups and determines whether one 
#tends to be higher or lower than the other. 

hot_prof = data_num[data_num["pepper"]==1]
not_hot_prof = data_num[data_num["pepper"]==1]

stat, p_val = mannwhitneyu(hot_prof["avg_rating"], not_hot_prof["avg_rating"])
print("Question 6")
print("U-stats: " ,stat)
print("p_value ", p_val)

sns.boxplot(x='pepper', y='avg_rating', data=data_num)
plt.title("Comparison of Average Ratings: Hot vs. Not Hot Professors")
plt.xlabel("Hot Professor")
plt.ylabel("Average Rating")
plt.xticks([0, 1], ['Not Hot', 'Hot'])
plt.show()
print("")


#Question 7 
#build a regression model predicting avg rating from difficulty only. 
#make sure to include the R^2 and RMSE of this model. 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error 

#convert avgf-diff into 2D array
avg_diff_reshaped = avg_diff.values.reshape(-1, 1)
#split data
X_train, X_test, y_train, y_test = train_test_split(avg_diff_reshaped, avg_rating, test_size = 0.2, random_state=random.seed(n_number))
#linear regression model
model = LinearRegression()
#prediction on the test set
model.fit(X_train, y_train)
#make predictions on the test set
y_pred = model.predict(X_test)
#evaluate the model 
mse = mean_squared_error(y_test, y_pred)
print("Question 7")
print(f"Mean Squared Error: {mse:.2f}")
plt.figure(figsize=(10, 6))

# Scatter plot of training data
plt.scatter(X_train, y_train, color='blue', label='Training Data', alpha=0.5)

# Scatter plot of test data
plt.scatter(X_test, y_test, color='pink', label='Test Data', alpha=0.5)

# Plot regression line
x_grid = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)
y_grid = model.predict(x_grid)
plt.plot(x_grid, y_grid, color='green', label='Regression Line')

# Labels and legend
plt.title('Regression Model: Average Rating vs. Difficulty')
plt.xlabel('Average Difficulty')
plt.ylabel('Average Rating')
plt.legend()
plt.grid(True)
plt.show()
print()

#question 8
#Build a regression model predicting avg rating from all available factors. 
#make sure to include the R^2 and RMSE of this model
#comment on how this model compares to the "difficulty only" model and on individual betas
#Make sure to address collinearity concerns. 
from sklearn.metrics import mean_squared_error, mean_squared_error, root_mean_squared_error
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

data_num = data_num.replace([np.inf, -np.inf], np.nan)
data_num = data_num.dropna()
X = data_num[["avg_difficulty","num_of_ratings", "pepper", "take_again_proportion","num_online","male","female"]]
y = data_num["avg_rating"]

#full model
X = sm.add_constant(X)
model_full = sm.OLS(y, X).fit()
y_pred_full = model_full.predict(X)

rmse_full = mean_squared_error(y, y_pred_full, squared=False)
r2_full = model_full.rsquared

#check for multicollinearity using VIF 
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print("Question 8")
print(model_full.summary())
print(f"\nR² (Full Model): {r2_full}")
print(f"RMSE (Full Model): {rmse_full}")
print("\nVIF Data (Multicollinearity Check):")
print(vif_data)
print()

sns.set(style="whitegrid")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(y, y_pred_full, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)  # Diagonal line for perfect prediction
plt.title('Predicted vs Actual Ratings')
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.xlim(y.min(), y.max())
plt.ylim(y.min(), y.max())


plt.show()


#9. build a classification model that predicts whether a professor
#receives a pepper from average rating only. <ake sure the include quality metrics suchs as AUROC and also address class imbalances. 
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
import matplotlib.pyplot as plt

data_num = data_num.replace([np.inf, -np.inf], np.nan)  # Replace infinite values with NaN
data_num = data_num.dropna(subset=["avg_rating", "pepper"])

X1 = data_num[["avg_rating"]]
y1 = data_num["pepper"]

X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=random.seed(n_number), stratify=y1)

#class imbalance 
smote = SMOTE(random_state=random.seed(n_number))
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

#logistic regression model 
model = LogisticRegression()
model.fit(X_train_res, y_train_res)

#predictions 
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

#Metrics 
auc = roc_auc_score(y_test, y_pred_proba)
print("Question 9")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(f"AUROC: {auc:.2f}")

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

#Question 10 

data_num = data_num.replace([np.inf, -np.inf], np.nan)  # Replace infinite values with NaN
data_num = data_num.dropna(subset=["avg_rating", "pepper", "avg_difficulty", "num_of_ratings", "take_again_proportion", "num_online", "male", "female", "pepper"])

X_all = data_num[["avg_rating", "avg_difficulty", "num_of_ratings", "pepper", "take_again_proportion", "num_online", "male", "female"]]
y_all = data_num["pepper"]

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=random.seed(n_number), stratify=y_all)

smote = SMOTE(random_state=random.seed(n_number))
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

model_all = LogisticRegression()
model_all.fit(X_train_res, y_train_res)

y_pred_all = model_all.predict(X_test)
y_pred_proba_all = model_all.predict_proba(X_test)[:, 1] 

auc_all = roc_auc_score(y_test, y_pred_proba_all)
print("Question 10")
print("Classification Report (All Factors):\n", classification_report(y_test, y_pred_all))
print("Confusion Matrix (All Factors):\n", confusion_matrix(y_test, y_pred_all))
print(f"AUROC (All Factors): {auc_all:.2f}")

fpr_all, tpr_all, thresholds_all = roc_curve(y_test, y_pred_proba_all)
plt.figure(figsize=(8, 6))
plt.plot(fpr_all, tpr_all, label=f'Logistic Regression (AUC = {auc_all:.2f}) - All Factors')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (All Factors)')
plt.legend(loc='lower right')
plt.show()

#extra credit
import scipy.stats as stats
data_cleanedd1 = data_num.dropna(subset = ["num_of_ratings", "avg_rating"])
avg_ratingg =data_cleaned1["avg_rating"]
num_ratingg = data_cleaned1["num_of_ratings"]
correlation, p_value = stats.pearsonr(avg_ratingg, num_ratingg)
print("Extra Credit")
print("correlation: ", correlation)
print("p_value", p_value)
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(8, 6))
sns.scatterplot(x=num_ratingg, y=avg_ratingg)
plt.title('Number of Ratings vs. Average Ratings')
plt.xlabel('Number of ratings')
plt.ylabel('Average Ratings')
plt.show()
print("")









