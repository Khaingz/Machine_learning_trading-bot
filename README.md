# Machine_learning_trading_bot

Challenge 14

Description: 

In this Challenge, ywe are assume the role of a financial advisor at one of the top five financial advisory firms in the world. 

The speed of these transactions gave our firm a competitive advantage early on. But, people still need to specifically program these systems, which limits their ability to adapt to new data. we’re planning to improve the existing algorithmic trading systems and maintain the firm’s competitive advantage in the market. To do so, we’ll enhance the existing trading signals with machine learning algorithms that can adapt to new data.

# Technology

In this module, we will use the following libraries:

- Pandas

- Numpy

- hvplot

- Matplotlib

- scikit-learn

# Imports
import pandas as pd
import numpy as np
from pathlib import Path
import hvplot.pandas
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from pandas.tseries.offsets import DateOffset
from sklearn.metrics import classification_report

Instructions:

Use the starter code file to complete the steps that the instructions outline. The steps for this Challenge are divided into the following sections:

## Establish a Baseline Performance

Step 1. Import the OHLCV dataset into a Pandas DataFrame.

Step 2. Generate trading signals using short- and long-window SMA values.

Step 3. Split the data into training and testing datasets.

Step 4. Use the SVC classifier model from SKLearn's support vector machine (SVM) learning method to fit the training data and make predictions based on the testing data. Review the predictions.
#### From SVM, instantiate SVC classifier model instance
svm_model = svm.SVC()
 
#### Fit the model to the data using the training data
svm_model = svm_model.fit(X_train_scaled, y_train)
 
#### Use the testing data to make the model predictions
svm_pred = svm_model.predict(X_test_scaled)

#### Review the model's predicted values
svm_pred

Step 5. Review the classification report associated with the SVC model predictions.
#### Use a classification report to evaluate the model using the predictions and testing data
svm_testing_report = classification_report(y_test, svm_pred)

# Print the classification report
print(svm_testing_report)

Step 6. Create a predictions DataFrame that contains columns for “Predicted” values, “Actual Returns”, and “Strategy Returns”.
#### Create a new empty predictions DataFrame.
#### Create a predictions DataFrame
predictions_df = pd.DataFrame(index=X_test.index)

#### Add the SVM model predictions to the DataFrame
predictions_df['Predicted'] = svm_pred

#### Add the actual returns to the DataFrame
predictions_df['Actual Returns'] = signals_df['Actual Returns']

#### Add the strategy returns to the DataFrame
predictions_df['Strategy Returns'] = predictions_df['Actual Returns'] * predictions_df['Predicted']
#### Review the DataFrame
display(predictions_df.head())
display(predictions_df.tail())

Step 7. Create a cumulative return plot that shows the actual returns vs. the strategy returns. Save a PNG image of this plot. This will serve as a baseline against which to compare the effects of tuning the trading algorithm.
#### Plot the actual returns versus the strategy returns
(1+ predictions_df[['Actual Returns', 'Strategy Returns']]).cumprod().hvplot(title= "Strategy Returns vs Actual Returns using SVM Supervised Learning with Original Input")

# Image attachment 






## Create an Evaluation Report

Establish a Baseline Performance

# Image attachment












# Image attachment














The baseline performance of the model has an accuracy at around 0.55 and return of about 48%. The baseline performance uses a SVC classifier model and generates signals using crossing SMA value
with a SMA short window of 4-days and a SMA long window of 100 days. The baseline training data is 3 months from the beginning the dataset.

## Tune the Baseline Trading Algorithm

Step 1. Tune the training algorithm by adjusting the size of the training dataset. To do so, slice your data into different periods. Rerun the notebook with the updated parameters, and record the results in your README.md file. 

# Image








# Image








Question: What impact resulted from increasing or decreasing the training window?
Hint To adjust the size of the training dataset, you can use a different DateOffset value—for example, six months. Be aware that changing the size of the training dataset also affects the size of the testing dataset.

Answer: After Increasing the trained dataset to be 6 months, we saw the accuracy and return are increased when compared to the baseline model.
The new performance with the 6 months training has an accuracy at around 0.56 and return of about 84%.

- We can see increasing the training window it can be improved our model accuracy slightly and improve our cumulative returns significantly.
- We can aslo see decreasing the training window did not seem to have any effect on the model accuracy or cumulative returns when compared to the oringinal data.


Step 2. Tune the trading algorithm by adjusting the SMA input features. Adjust one or both of the windows for the algorithm. Rerun the notebook with the updated parameters, and record the results in your README.md file.

# Image






# Image





Question: What impact resulted from increasing or decreasing either or both of the SMA windows?

Answer : Decreasing the SMA short windows to 2 days rather than 4-days, we can see that has increased the accuracy but decreased the return compare to the baseline model. 

The new performance with the adjusting the SMA short window has an accuracy at around 0.56 and return of about 41%.

Step 3. Choose the set of parameters that best improved the trading algorithm returns. Save a PNG image of the cumulative product of the actual returns vs. the strategy returns, and document your conclusion in your README.md file.

The best improved trading algorithm returns

# Image









Conclusion : I found that increasing the training data set from 3 months to 6 months and maintaining the existing SMA wndows ( Shorts 4 days, Long 100 days) had the biggest infrease on returns when using the original SVC classifier model.After adjusting the input parameters, the new returns were around 84% compared to the original input parameters 48%.

## Evaluate a New Machine Learning Classifier

In this section, you’ll use the original parameters that the starter code provided. But, you’ll apply them to the performance of a second machine learning model. To do so, complete the following steps:

Import a new classifier, such as AdaBoost, DecisionTreeClassifier, or LogisticRegression. (For the full list of classifiers, refer to the Supervised learning page in the scikit-learn documentation.)
#### Import a new classifier from SKLearn
from sklearn.ensemble import AdaBoostClassifier 

#### Initiate the model instance
model = AdaBoostClassifier()

Using the original training data as the baseline model, fit another model with the new classifier.
#### Fit the model using the training data
model = model.fit(X_train_scaled, y_train)

#### Use the testing dataset to generate the predictions for the new model
pred = model.predict(X_test_scaled)

#### Review the model's predicted values
pred

Backtest the new model to evaluate its performance. Save a PNG image of the cumulative product of the actual returns vs. the strategy returns for this updated trading algorithm, and write your conclusions in your README.md file. 
#### Use a classification report to evaluate the model using the predictions and testing data
Ada_testing_report = classification_report(y_test, pred)

#### Print the classification report
print(Ada_testing_report)

#### Create a new empty predictions DataFrame.
#### Create a predictions DataFrame
predictions_df = pd.DataFrame(index=X_test.index)

#### Add the SVM model predictions to the DataFrame
predictions_df['Predicted'] = pred

#### Add the actual returns to the DataFrame
predictions_df['Actual Returns'] = signals_df['Actual Returns']

#### Add the strategy returns to the DataFrame
predictions_df['Strategy Returns'] = predictions_df['Actual Returns'] * predictions_df['Predicted']

#### Review the DataFrame
display(predictions_df.head())
display(predictions_df.tail())

#### Plot the actual returns versus the strategy returns
(1+ predictions_df[['Actual Returns', 'Strategy Returns']]).cumprod().hvplot(title= "Strategy Returns vs Actual Returns using AdaBoost Supervised Learning")


# Image








# Image











Questions: Did this new model perform better or worse than the provided baseline model? Did this new model perform better or worse than your tuned trading algorithm?

Answer : The new machine learning model has the same accuracy as the original model but has an increased return. The accuracy of the new model is aorund 0.55 and the return is around 57%.
The return is significant increase compared to the 48% return from the original model, which is indicates this new model may be better to utilize AdaBoost Supervised Learning.