# Real Estate Price Prediction for Madrid Municipality

This is the final project for the Kschoool Master in Data Science (4th ed), and it is focused on predicting real estate prices in the Madrid municipality. The goal of the project is to build a model that can accurately predict the prices of properties in Madrid based on various features, such as district, size, property type, etc.

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

These instructions will help you to download and run the project on your local machine for development and testing purposes.

### Prerequisites

Make sure you have Python 3.x installed and a https://scraperapi.com/ account.

### Installing

Clone the repository to your local machine using the following command:

```
$ git clone https://github.com/lorquez/real_estate_price_predictor.git
```

Then simply open a terminal in the project folder and type:
```
pip install -r requirements.txt
```

Finally, paste the scraperapi api key in the config.json file located in the config folder.

### Execution

Data can be obtained by running the two notebooks in the "/scr/0. Scraping/" folder. After data is obtained, run the notebooks from 1 to 6 in order to execute the project.
Access this url to see the final Streamlit: https://realestate-prediction.herokuapp.com/

## Author

In alphabetic order:
- Alessandro cosci (alick888@gmail.com)
- Miguel Ángel Alberola (mangel.alb@gmail.com)

## License

This project is licensed under the MIT License.
