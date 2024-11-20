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
The E-Commerce Fraud data set, found here https://www.kaggle.com/datasets/vbinh002/fraud-ecommerce/data, 
is a Kaggle dataset and contains simulated credit card transactions where fraudulent 
transactions are scattered through-out the data set. The data contains simulated
transactions from 01/01/2015 to 12/16/2015 from different browsers, sources, and 
IP addresses. The data set is a 16.1 MB structured data set that contains 151,112 rows and 11 columns. 
Thedata set is very clean and contains no missing or duplicate values, making it
suitable for analysis.
## Data Collection
The data set was downloaded directly from Kaggle in CSV format. Once down-
loaded, the data set was placed in the correct folder for work within VS Code,
saved in VS Code, and then pushed to GitHub. The process did not require any
special scraping or data extraction techniques.
## Data Dictionary
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
