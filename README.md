# Capstone-Miller
## E-Commerce Fraud Detection Model

## Description
The goal of this project is to create a model that can predict fraudulent E-Commerce credit card transactions. The project utilizes simulated credit card data to explore the ability of machine learning to identify fraudulent E-Commerce credit card transactions. Success in this project could suggest that machine learning can be used to identify E-Commerce fraud, and could be used to save businesses and individuals from significant financial loss. 

## Project start
1. Start a new repository and select default README.md
2. Clone the repository to local environment. I Used VS Code to clone the repository. 
3. Open the project, create, and activate the virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```
4. Install dependencies into the .venv
```bash
python3 -m pip install pandas scikit-learn seaborn matplotlib
```
5. Freeze dependencies into requirements.txt
 ```bash
 python3 -m pip freeze > requirements.txt
  ```
## Project Introduction
E-Commerce has experienced great growth over the last several years as a result
of accessibility and convenience, especially during pandemic lockdowns [4]. How-
ever, the growth of E-Commerce has also led to the growth of E-Commerce fraud.
E-Commerce fraud refers to unauthorized activity that exploits on-line shopping
platforms and payments where fraudsters have access to personal information
and make financial gains [5]. According to Juniper Research, E-Commerce fraud
is expected to increase from $44.3 billion in 2024 to $107 billion in 2029 [6].
Types of E-Commerce fraud include card-not-present fraud, stolen credit card
fraud, friendly fraud, identity fraud, card testing fraud, refund abuse, and trian-
gulation fraud [1]. Fraudulent transactions are harmful both to businesses and
individuals, as they can lead to significant financial losses.
Detecting and stopping E-Commerce fraud is crucial as E-Commerce con-
tinues to grow. Machine learning can be used to analyze data, identify pat-
terns, and adapt to trends to improve fraud detection [2]. This study aims to
develop a model that correctly identifies card-not-present fraud in the Kaggle
dataset, https://www.kaggle.com/datasets/vbinh002/fraud-ecommerce/data, of
simulated E-Commerce transactions, as real transactions cannot be used due to
privacy issues

## Project Implementation Steps
1. Introduce the problem and define the objective
(a) Introduction and Goals
(b) Project Limitations
2. Data Collection
(a) Data Source
(b) Data Description
(c) Data Attributes
3. Prepare Data
(a) Data Cleaning
(b) Feature Engineering
4. Conduct exploratory data analysis (EDA).
(a) Descriptive Statistics
(b) Initial Visualization
5. Select, train, and evaluate models
(a) Select Models
(b) Train Models
(c) Evaluate and Compare Models
6. Results and Conclusions
(a) Interpretation of Results
(b) Conclusions and Insights

## Goals
1. Ensure the use of clean data and create a new feature to better evaluate the
data.
2. Analyze fraudulent transactions to determine which features predict fraud
with the highest precision and accuracy.
3. Evaluate multiple models to determine which model best predicts fraudulent
E-Commerce transactions.
## Limitations
– Use of Simulated Data: Simulated data is used to protect the privacy of
customers. Simulated data can lack complexity and may show bias, so it
would not be as affective as real data.
– Generalization or Overfitting: The simulated data is generalized to one type
of fraudulent transaction, card-not-present fraud, and would likely struggle
to identify other types of fraud. The simulated data could also be based on
previous fraud patterns, so as new fraud patterns emerge, the model would
be less effective in detecting fraud.
## Data Collection
### Data Source and Description
The E-Commerce Fraud data set, found here https://www.kaggle.com/datasets/vbinh002/fraud-ecommerce/data, 
is a Kaggle dataset and contains simulated credit card transactions where fraudulent 
transactions are scattered through-out the data set. The data contains simulated
transactions from 01/01/2015 to 12/16/2015 from different browsers, sources, and 
IP addresses. The data set is a 16.1 MB structured data set that contains 151,112 rows and 11 columns. 
Thedata set is very clean and contains no missing or duplicate values, making it
suitable for analysis.
### Data Collection
The data set was downloaded directly from Kaggle in CSV format. Once down-
loaded, the data set was placed in the correct folder for work within VS Code,
saved in VS Code, and then pushed to GitHub. The process did not require any
special scraping or data extraction techniques.
### Data Dictionary
| Column Name      | Description                             | Data Type | Example Value        |
|------------------|-----------------------------------------|-----------|----------------------|
| Age              | Age of the user                        | int64     | 39                   |
| Browser          | Browser used                            | object    | Chrome               |
| Class            | Fraud label (0 or 1)                   | int64     | 0                    |
| Device ID        | Unique device identifier               | object    | QVPSPJUOCKZAR        |
| IP Address       | IP address used during transaction     | float64   | 732758368.79972      |
| Purchase Time    | Time of Initial Purchase               | object    | 2015-04-18 02:47:11  |
| Purchase Value   | Value of the purchase transaction      | int64     | 34.00                |
| Sex              | Gender of the user                     | object    | M                    |
| Signup Time      | Time of Initial Signup                 | object    | 2015-02-24 22:55:49  |
| Source           | Source of traffic                      | object    | SEO                  |
| User ID          | Unique identifier for each user        | int64     | 22058                |
## Clean and Process Data
### Data Cleaning and Feature Engineering
A Jupyter Notebook within VS Code was used to check the data for any nec-
essary cleaning. The Pandas library was then used to check for missing or du-
plicated values. There were no duplicated or missing records in the dataset, so
these issues did not need to be addressed. Pandas was also used to check the
data type of each column to ensure that each column had the correct data type.
The columns signup time and purchase time were object data type and had
to be converted to date-time data type.
### Feature Engineering
Once signup time and purchase time were converted to date-time data, fea-
ture engineering was used to create a new column for analysis. For the new
column, signup time time was subtracted from purchase time to represent the
time between sign-up and initial purchase. This new column could help identify
fraudulent purchases where there is little time between the sign-up time and the
initial purchase time. This new column eliminates the need for the signup time
and purchase time columns, so Pandas was used to drop both columns from
the dataset. The cleaned and transformed data set was then saved as a new CSV
using Pandas.
### Cleaned and Feature Engineered Data Dictionary
In the cleaned dataset, columns user id, purchase value, device id, source,
browser, sex, age, ip address, and time difference will be independent vari-
ables. The column class, that classifies a transaction as fraudulent or non-
fraudulent, is the dependent variable.
| Column Name      | Description                                    | Data Type | Example Value        |
|------------------|------------------------------------------------|-----------|----------------------|
| User ID          | Unique identifier for each user               | int64     | 22058                |
| Purchase Value   | Value of the purchase transaction              | int64     | 34.00                |
| Device ID        | Unique device identifier                       | object    | QVPSPJUOCKZAR        |
| Source           | Source of traffic                              | object    | SEO                  |
| Browser          | Browser used                                    | object    | Chrome               |
| Sex              | Gender of the user                             | object    | M                    |
| Age              | Age of the user                                | int64     | 39                   |
| IP Address       | IP address used during transaction             | float64   | 732758368.79972      |
| Time Difference  | Time Between Sign-up and Purchase (Minutes)    | float64   | 75391.4              |
| Class            | Fraud label (0 or 1)                           | int64     | 0                    |
## Exploratory Data Analysis
Exploratory Data Analysis is an important part of a data analytics project [3].
EDA can be used to identify patterns, anomalies, relationships, and insights
from the data [3]. Unlike other methods that are used to confirm a hypothesis,
EDA allows you to generate new hypotheses from the data [3]. EDA ensures
that data are objectively assessed and can be used to thoroughly describe data
before performing more complex analyses [3]. EDA in the early stages of a project
can ensure the quality of the data and help fit the data to the correct model,
maximizing potential insights [3]

EDA for the project was performed in a Jupyter notebook within VS Code,
and the notebooks can be found here https://github.com/gmill88/Capstone-Miller/
blob/main/EcommerceFraudEDA.ipynb. The cleaned data set was first loaded
into the notebook; then important Python libraries such as pandas, seaborn,
matplotlib, and scikit-learn were imported. Commands like head(), .shape, and
info() were used to inspect the cleaned data. These commands provide a quick
look at the first few rows of data, the number of rows and columns in the dataset,
the number of non-null rows, and the datatype of each row in the data. These
commands revealed that there are 10 columns and 151,112 rows in the cleaned
data, and all rows contain non-null data (no missing values). After inspection,
describe(include=’all’) was used to generate summary statistics for numerical
columns and frequency statistics for the categorical columns. A bar graph was
used to inspect the distribution of class throughout the dataset, as class is the
variable the model will attempt to predict. Box plots, bar graphs, and vio-
lin graphs were then used to inspect the relationship of the class with other
columns like purchase value, source, browser, sex, and time difference.
A correlation heat map was then used to inspect the correlation between age,
class, purchase value, time difference, and class in an attempt to deter-
mine how correlated these variables were and if there were any strong correlations
to avoid when creating a model.
### Descriptive Statistics
![E-Commerce Fraud Dataset Descriptive Statistics](Images/DescriptiveStats.png)

Figure 1: E-Commerce Fraud Dataset Descriptive Statistics

## Data Visualizations

Since the goal of the project is to create a model to predict fraudulent transactions, the EDA is focused on `class` which indicates whether or not a transaction was fraudulent. It is important to determine the distribution of fraudulent transactions because these transactions are rare, so this is important information to keep in mind when evaluating a model's performance.

### Class Distribution: Non-Fraud vs. Fraud

![Class Distribution Non-Fraud vs. Fraud](Images/image.png)

_Figure 1: Class Distribution Non-Fraud vs. Fraud_

Since there is an obvious class imbalance between fraudulent and non-fraudulent transactions, it is important to use metrics other than just accuracy to evaluate a model. Metrics like precision, recall, and F1 score will do a better job of indicating the performance of a model on the minority class. Understanding that there is a class imbalance for fraudulent and non-fraudulent transactions can also help pick a model that better handles a class imbalance.

---

After understanding the distribution of `class`, it is important to compare `class` to other variables to determine if a relationship exists between the two.

### Purchase Value by Class

![Purchase Value By Class](Images/image2.png)

_Figure 2: Purchase Value By Class_

The "Class vs. Purchase Value" figure shows that there is no significant difference in purchase value between fraudulent and non-fraudulent transactions. Therefore, purchase value will not be a good predictor of fraudulent purchases, and other features should be explored.

---

### Fraudulent Transaction Percentage by Source

![Fraudulent Transaction Percentage by Source](Images/image3.png)

_Figure 3: Fraudulent Transaction Percentage by Source_

The "Source vs. Fraud Percentage" bar graph displays the percentage of transactions generated from each source that were fraudulent. Direct transactions had a higher fraud percentage than transactions generated from ads or SEO (search engine optimization). While direct transactions had a higher percentage of fraudulent transactions, it was only around 1.5% higher than the other sources, so its importance in modeling will need to be further explored.

---

### Fraudulent Transaction Percentage by Browser

![Fraudulent Transaction Percentage by Browser](Images/image4.png)

_Figure 4: Fraudulent Transaction Percentage by Browser_

The "Browser vs. Fraud Percentage" bar graph shows the percentage of transactions by browser that were fraudulent. Chrome and Firefox had a higher percentage of fraudulent transactions than the other browsers, but the difference in percentage between each browser is small. That being said, it is unlikely that the browser used will be a good indicator of fraudulent transactions.

---

### Fraudulent Transaction Percentage by Sex

![Fraudulent Transaction Percentage by Sex](Images/image5.png)

_Figure 5: Fraudulent Transaction Percentage by Sex_

The "Sex vs. Fraud" bar graph is used to determine if one sex was more likely to complete a fraudulent transaction than the other. The graph shows the percentage of the transactions that were fraudulent for both sexes represented in the data. Fraudulent transactions by males were 0.45% higher than females. While there is a difference in the percentage of fraudulent transactions between the sexes, the difference is not enough to make sex a good indicator of fraudulent transactions.

---

### Time Difference Between Signup and Initial Purchase for Non-Fraudulent and Fraudulent Transactions

![Violin Plot of Time Difference Distributions for Non-Fraudulent and Fraudulent Transactions](Images/image6.png)

_Figure 6: Violin Plot of Time Difference Distributions for Non-Fraudulent and Fraudulent Transactions_

The violin plot of the time difference distributions for non-fraudulent and fraudulent transactions is used to identify patterns in how quickly an initial purchase is made after sign-up. Based on the plot, non-fraudulent transactions have a uniform spread, so there are no discernible patterns in time difference for non-fraudulent transactions. Fraudulent transactions, however, are more concentrated around zero, meaning fraudulent transactions have significantly lower time difference values than non-fraudulent transactions. The difference in distribution of time differences between fraudulent and non-fraudulent transactions looks to be significant and could be a good indicator for fraudulent transactions.

---

### Fraudulent and Non-Fraudulent Transactions Based on Device User Type

![Grouped Bar Chart of Devices as Single or Multiple Users for Fraudulent and Non-Fraudulent Transactions](Images/image7.png)

_Figure 7: Grouped Bar Chart of Devices as Single or Multiple Users for Fraudulent and Non-Fraudulent Transactions_

The grouped bar chart displaying devices as single or multiple users for fraudulent and non-fraudulent transactions can be used to identify patterns in fraudulent transactions for devices with a single user or multiple users. For fraudulent transactions, devices with multiple users accounted for nearly 60% of the fraudulent transactions. For non-fraudulent transactions, less than 5% of transactions occurred on a device with multiple users. This shows that the majority of non-fraudulent transactions occur on a device with a single user, while a large percentage of fraudulent transactions occurred on a device that used multiple user IDs. Therefore, a `device_id` with multiple `user_id`'s could be a good indicator of fraudulent transactions.

---

### Numerical Data Correlation Heatmap

![Correlation Heatmap](Images/image8.png)

_Figure 8: Correlation Heatmap_

The heatmap shows the correlation between numerical features in the data. All features are almost completely uncorrelated, with the exception of `time_difference` and `class`, which are weakly correlated. For the sake of creating a model, weak to no correlation is preferred. Strongly correlated features can reduce a model's accuracy and lead to redundancy within the model, causing overfitting.

---

## Insights and Observations From Exploratory Data Analysis

Several key insights were gained from the exploratory data analysis portion of the project. Visualization of the class distribution revealed a class imbalance, so it is important to include metrics such as precision, recall, and the F1 score to evaluate model performance. Other visualizations found that the `purchase_value`, `source`, and `browser` are not good indicators of class, but including these features in the model could lead to higher predictive power. No particular browsers or sources were attributed to a higher percentage of fraudulent transactions, and differences in purchase value were not good indicators of fraudulent purchases. 

`Sex` was also a poor indicator of fraudulent transactions, as the percentage of fraudulent transactions is nearly the same for males and females. `Time_difference` was the first feature found to be a possible indicator of fraudulent transactions. Fraudulent transactions were found to have a shorter time difference between initial sign-up and purchase, so `time_difference` could be a valuable feature for a fraud detection model. 

The type of device user was also found to be a possible indicator of fraudulent transactions. Devices with multiple users accounted for a higher percentage of fraudulent transactions compared to non-fraudulent transactions. This suggests that multiple users could be an indicator of fraudulent activity. Lastly, a heatmap was utilized to determine the correlations between numerical features. The correlation of features was found to be relatively weak, suggesting that these variables will not cause redundancy or overfitting of the model.

Based on the EDA, the features `time_difference` and `user_id` are likely the best to include in a model that predicts fraudulent transactions. Both features appear to be indicators of fraudulent transactions, and therefore these features will be useful in creating a model that predicts fraudulent transactions.

