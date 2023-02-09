# Real Estate Price Prediction for Madrid Municipality

This is the final project for the Kschoool Master in Data Science (4th ed), and it is focused on predicting real estate prices in the Madrid municipality. The goal of the project is to build a model that can accurately predict the prices of properties in Madrid based on various features, such as district, size, property tye, etc.

## Objective

This project comes with the objective of knowing the ideal market prices of second-hand homes in the municipality of Madrid. To do so, the model analyzes a dataset of real estate properties. Given a specific set of property's characteristics, the model is able to estimate its ideal price range.

![Streamlit front-end](https://github.com/lorquez/real_estate_price_predictor/blob/main/raw_data/TFM_frontend.jpeg)

## Project Overview

The project consists of the following steps (one per each notebook):

0. Data collection: scraping real estate data from idealista.com and fotocasa.es using Selenium and scraperapi.
1. Data loading and understanding: A first data cleaning is performed to eliminate scraping issues and merge the 2 datasets into one
2. Data cleaning: A deeper cleaning of the merged dataset is performed to prepare the data for the EDA.
3. Exploratory data analysis: Analizing the cleaned data to gain insights into the relationships between the various features and the target variable (property prices) as well as to spot outliers.
4. Feature engineering: Preparing the data for modeling.
5. First model building: Building and training of several machine learning models to predict real estate prices and see which performs better.
6. Model tuning evaluation: Evaluation of the model's performance using various metrics and making necessary improvements through hyperparameters' tuning.

## Getting Started

These instructions will help run the project for development and testing purposes.

### Prerequisites

Make sure you have Python 3.x installed to run the Fotocasa scraping notebook and a https://scraperapi.com/ account to run the Idealista scraping notebook.

### Installing

Besides "0. Fotocasa Scraping (Jupyter Notebook).ipynb" all the notebooks are meant to be executed on Google Colab. To do so, just clone the repository and upload the whole set of folders and files inside a folder called "Kschool_TFM" into your Google Drive.

Alternatively, you can download the project folder from [here](https://drive.google.com/drive/folders/1fTTUgWePBQj0mdQUhksbH02B9dyaU3O1?usp=share_link) and then upload it in your Google Drive into a folder called "Kschool_TFM". This is the best way as it already includes the CSVs that had been created during the project development.

To clone the repository to your local machine use the following command:

```
$ git clone https://github.com/lorquez/real_estate_price_predictor.git
```

Now that the files are on your local machine, paste the scraperapi api key in the config.json file located in the config folder.

Finally upload the data as metioned above.

### Execution

Data can be obtained by running the two notebooks in the "/scr/0. Scraping/" folder. To run the "0. Fotocasa Scraping (Jupyter Notebook).ipynb" you have to download the notebook on your local machine and simply run the cells in order. After data is obtained, run the notebooks in the "/src/" from 1 to 6 in order to execute the project.
Access this url to see the final Streamlit: https://realestate-prediction.herokuapp.com/

## Authors

In alphabetic order:
- Alessandro cosci (alick888@gmail.com)
- Miguel Ángel Alberola (mangel.alb@gmail.com)
