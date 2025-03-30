# Pharama-sales-prediction

#### **Introduction**

Here’s an overall idea for your **Pharmacy Sales Prediction System** project, formatted for your GitHub description:  

---

# 📊 Pharmacy Sales Prediction System  

## 🔥 Overview  
This project is a **machine learning-based forecasting system** designed to predict **weekly pharmacy sales** based on historical sales data. It allows users to input a **category** (e.g., medicine type) and a **date**, returning the closest valid forecasted sales value.  

## 🎯 Key Features  
- **Historical Data Processing**: Cleans and processes past pharmacy sales records.  
- **Date Modification**: Adjusts dataset dates (e.g., shifting years to 2019).  
- **Sales Forecasting**: Predicts future sales using machine learning models (e.g., ARIMA, LSTM, or XGBoost).  
- **User Input Handling**: Accepts user-specified **category** and **date** to return the most relevant prediction.  
- **Visualization**: Displays trends using **matplotlib/seaborn** for better insights.  

## 🏗️ Tech Stack  
- **Python (Pandas, NumPy, Scikit-learn, TensorFlow/PyTorch)**  
- **Machine Learning (Time Series Models: ARIMA, LSTM, XGBoost, etc.)**  
- **Data Visualization (Matplotlib, Seaborn)**  
- **Flask/FastAPI (Optional, for API deployment)**  

## 🚀 How It Works  
1. **Preprocess Data** – Load, clean, and modify date formats if necessary.  
2. **Train Model** – Use historical sales data to train a predictive model.  
3. **User Input** – Accept a **category** and a **date** as input.  
4. **Find Closest Valid Prediction** – Return forecasted sales for the nearest valid date.  
5. **Visualize & Interpret Results** – Show trends and patterns.  

## 📌 Future Improvements  
- Support for **multiple pharmacies** and **seasonal trends**.  
- Web-based **interactive UI** for easy predictions.  
- Integration with **real-time data sources** for live forecasting.  

---


* **Background of the problem**

    Pharmacists and drug sellers in Sri Lanka face many challenges due to issues likeweather, importing difficulties, andeconomic problems. This often leads to a shortage of medicines when  people need them most. To help, we want to create a data science system that predicts these issues, so sellers can better prepare and reduce shortage.

#### Design 

---

![1730391247387](image/README/1730391247387.png)

#### Implementation

---

**Technologies**

* Frontend development - HTML,CSS and Flask render templates
* Backend development - Python,Pandas,Matplotlib,adfuller,Flask

**Data collection and pre-processing**

* We gathered data from two pharmacies in particular area.
* Then categorize those data according to ATC classification system managed by WHO (e.g., acetaminophen belongs to ATC code -N02BE01).

  ![1730391858536](image/README/1730391858536.png)
* We will process the dataset by handling null values, addressing missing data, and managing outliers accordingly.

**Model Implementation**

* Reason for use time series forecasting:
  1.SARIMA handles seasonality
  2.Captures trends and seasonality in sales
  3.Provides reliable forecast for higher accuracy
* We split entire dataset into Train and Test data.

**Testing and Evaluation**

*
    Testing for C1*

![1730392487570](image/README/1730392487570.png)

*
    Accuracy for each category*

![1730392533836](image/README/1730392533836.png)

* Augmented Dickey-Fuller (ADF) Test used to check if the sales data was stationary (required for time-series forecasting).ADF test showed non-stationary data (p-value > 0.05).
* Applied differencing to make the data stationary.
* Chosen for its ability to handle seasonal patterns in the sales data.
* Compared predicted vs. actual test sale data, showing reliable and accurate forecasts for drug categories.
* The model helped pharmacists manage inventories, reducing shortages and wastage.
