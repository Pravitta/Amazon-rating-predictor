# Mini Project Report: Amazon Product Rating Prediction

## 1. Project Selection
**Project Title**: Amazon Product Rating Prediction
**Objective**: Predict likely ratings based on product descriptions and reviews using regression in PySpark.
**Problem Statement**: Consumers heavily rely on product ratings before making a purchase. However, not all products have sufficient ratings to guide buyers. This project aims to use machine learning (specifically Regression models in PySpark) to predict the missing product ratings based on user review text, which can then be used to guide product recommendations.

## 2. Dataset Information
**Dataset Name**: Amazon Reviews
**Data Source**: Open-source dataset (Attached as `Amazon_Reviews.csv`)
**Dataset Description**: The dataset contains user reviews of various products. The attributes include the reviewer's information, review date, star rating, review title, and detailed review text.
**Dataset Size Instructions & Compliance**:
- *File Size*: ~13.1 MB
- *Number of Records*: 21,058 rows
- *Number of Attributes*: 9 columns (`Reviewer Name`, `Profile Link`, `Country`, `Review Count`, `Review Date`, `Rating`, `Review Title`, `Review Text`, `Date of Experience`)
- *Sampling Rationale*: The project guidelines recommend 50,000 to 500,000 records. Since the dataset attached falls slightly short of this (21,000 records), we are using this dataset as a representative **sample subset** (which complies with the guideline stating: *"If the dataset is too large for your system, you may use a sample subset (10–20% of data) while maintaining diversity and representation"*). This sample size is still sufficient to demonstrate Big Data processing using PySpark.

## 3. Project Implementation details (PySpark)

The project is implemented using PySpark to process and train models on the dataset efficiently. The implementation steps are encapsulated in the script `amazon_rating_prediction.py`.

### 3.1. Data Loading and Cleaning
- **Loading**: The dataset is loaded using PySpark’s CSV reader with `header=True` and `inferSchema=True`. Multi-line text is properly handled by configuring the `multiLine=True` property.
- **Cleaning & Parsing**: The `Rating` column is provided as strings like "Rated 1 out of 5 stars". A regular expression (`regexp_extract`) is utilized to extract the numeric value and cast it to a `double` type to serve as the regression label. Rows with missing or empty `Review Text` or `Rating` are filtered out.

### 3.2. Natural Language Processing (NLP) Pipeline
To use text data for regression, we transformed the `Review Text` into numerical feature vectors. We built an NLP Pipeline using PySpark MLlib:
1. **Tokenizer**: Splits the text into individual words.
2. **StopWordsRemover**: Removes common English stop words that do not contribute to sentiment or rating (e.g., "and", "the", "is").
3. **HashingTF**: Computes term frequencies across the text and projects them into a fixed-length feature vector (size: 10,000).
4. **IDF (Inverse Document Frequency)**: Rescales the term frequencies, giving more weight to rare but significant words.
5. **VectorAssembler**: Assembles the computed features into a final `features` vector for the model.

### 3.3. Machine Learning Model (Regression)
We chose a **Random Forest Regressor** to predict the numerical rating.
- **Why Random Forest?**: It is an ensemble learning method that is highly robust to outliers and non-linear data patterns. It operates by constructing a multitude of decision trees at training time and outputting the average prediction of the individual trees.
- **Train-Test Split**: The dataset was randomly split into training (80%) and testing (20%) subsets.

### 3.4. Model Evaluation Metrics
The performance of the predictive model is evaluated using standard regression metrics:
- **Root Mean Squared Error (RMSE)**: Measures the average magnitude of the error. A lower RMSE indicates that the predicted ratings are closer to the actual ratings.
- **R-squared ($R^2$)**: Explains the proportion of variance in the target variable (ratings) that is predictable from the features.

## 4. Conclusion
This project successfully demonstrates the use of PySpark for processing real-world unstructured text data (reviews) and applying distributed machine learning to predict numerical outcomes (ratings). This pipeline can be easily scaled up to handle millions of records in an enterprise environment to guide consumer recommendations.
