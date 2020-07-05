# Tricentis - Lead Scoring Project

Complete repositiory for the Georgia Tech Capstone Project hosted by Tricentis - a modern DevOps and test driven development solutions company. 

[Tricentis](https://www.tricentis.com/)

Project midterm report and final are included in the `references` directory. 

  - [Capstone Final Report](https://github.com/olivierzach/tricentis_lead_scoring/blob/master/references/Capstone%20Final%20Report.pdf)
  
  - [Capstone Midterm Report](https://github.com/olivierzach/tricentis_lead_scoring/blob/master/references/progress_report.pdf)


# Project Goals

Goal of this project is to develop a machine learning system that scores leads likelihood to convert based on interations within the marketing-sales funnel. Model scores determine which leads are passed through to the sales team to to start a product engagement. 

Solution solves key business problems:
  - Filter out bad leads that take up sales resources, saving time and effort to focus on best leads
  - Identifies the best leads for the sales team to focus on, ensures best resources are working on best deals
  - Provides inference into what factors drive leads to a higher conversion likelihood, allows marketing to organically grow these leads 
  - Scores serve as the base for website optimizaion and retargeting to increase form fills of the most qualified leads
  - Solution is set up to be hosted online for real time scoring

Final solution is an end-to-end data science project that comes together in these steps:
  - Data Sourcing
  - Data Cleaning and Feature Engineering
  - Exploratory Data Analysis
  - Model experiments, validation, and model selection
  - Template for hosting model for online scoring
  - Business Recommendations

# Data

Available data for the project includes over 1000+ variables related to the marketing-sales funnel. 

  - `Marketing`: data related to the marketing channel a lead originated from (i.e. Facebook, Google)
  - `Website Touchpoints`: all iteractions with the Tricentis website including clicks, session length, form fill, content downloads, bounces etc.
  - `CRM Touchpoints`: retargeting campaigns, lead activity, sales status, lead segmentation
  - `Enrichment Data`: appended attributes to the lead record, includes user and business attributes

Data processing steps include:
  - High cardinality categorical variable encoding
  - Categorical Variables Encoding
  - Time Based engagement features
  - Missing Value imputation
  - Channel, URL, Content path including order of touchpoints through time (i.e. encode the customer journey through the funnel: Facebook > Demo Page > Resource Page > Free Trial > Form Fill, capturing the order of this path)
  - URL, Content segmentation to enable forward compatability of model solution using TF, TF-IDF
  - Target Definition
  - Common join key across all data = connection the entire marketing-sales funnel data sources together
  - Handling censoring to prevent model leakage
  - Handling extremely imbalanced data (accepted sales leads ~1% of total leads)
  - Scaling features before model experiements
  - Sparsity Reduction through regularization and near zero variance filters

All data assets are included in the `data` directory. Here are quick descriptions:

  - `data_factory.py`: functions to generate and serialize clean, feature rich datasets
  - `profile_{dataset}.py`: data cleaning and feature engineering functions
  - `train_factory.py`: script to generate model training data and serialize
  - `data_util.py`: helper analysis and data cleaning functions
  - `pickle`: save analysis and model objects
  - `eda_output`: visualizations for data profiling and analysis

**Note: Project is governed by a NDA that restricts sharing original data source.**

# Model Experiments

Following the `No Free Lunch` theorem mutiple models (parametric and non-parametric) were tested and evaluated for classifying leads as accepted or not accepted. 

All models were developed (hyperparameter tuning) and evaluated using a simple 5-fold cross validation strategy on a subset of the training data. Models were evaluated across a host of classification metrics including: `Accuracy`, `Confusion Matrix`, `Precision`, `Recall`, `F1 Score`, and `Percent Leads Predicted Accepted`. The classification threshold was analyzed at each step to optimize the tradeoff between precsision and recall by plotting these curves across threshold cutoffs. 

Best model chosen was then fit to a hold out dataset to confirm results. Final selected model was then trained on all available data and serialized for production. 

Models Tested:
  - Regularized Logistic Regression (Lasso penalty)
  - Extra Trees Classifer
  - Random Forest
  - Boosted Trees Classifier
  - Deep Neural Network

Scripts to develop and test these model available in the `model` directory. 

# API

Flask template to automate the scoring process was developed and is included in the `app` directory. 

Template sets up a simple API to respond to a post request with the model's probabalistic score of sales acceptance. Dockerfile available for encapsulating dependencies and hosting on a Tricentis specific platform. 

Request route flow: 
  - Take in and parse incoming request with lead id and precomputed features
  - Load model assets: best model, scaler learned from training, and optimal classification threshold
  - Scale the input data
  - Predict probability scores and accepted flag based on threshold
  - Tag scores with a score date
  - Return the scores to enable downstreaming routing of leads to sales team through CRM

# Results

Of all models the best to balance precision and recall was the `Boosted Trees Model`. Other models performed well but struggled with low precision scores. Linear and Tree based models were still valuable for inference of key variables through coefficient estimates and feature importance.  

Final Model Results:
  - `ROC-AUC`: .86
  - `Recall`: .75
  - `Precision`: .22 (almost 5x precision of other models)
  - `Accuracy`: .97
  - `F1 Score`: .98
  - `Percent Leads Predicted Accepted`: 3% (baseline rate 1%)
 

# Business Implications

Insights are available from all steps of the modeling process, not just the final model itself.

  - Require complete form fills to boost data collection
  - **Randomly send leads to the floor to judge performance of incumbant model, this will allow future models to learn true relationships not just the previous model's rules**
  - Develop more custom content and website assets to engage more website visitors with actual activities, inference indicates activity matters more than enrichment qualification
  - Develop a way to collect data on "offline" leads to enable scoring - "offline" leads have no connected web activity
  - **Use lead scores to grow qualified lead form fills - high scored web visits with no form fill should be heavily retargeted**
  - Attributes driven segmentation of URLs, Content, including deeper NLP analysis to group these dynamic featuring into future-proof buckets, helps ensure success of model and future model efforts
  - **Host the final model developed here to filter leads to the sales team: model performance shows the final model best balances the costs of sending too many leads versus holding out strong leads**
