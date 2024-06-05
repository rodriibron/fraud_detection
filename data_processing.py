
import pandas as pd
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



### class DataProcessing ###
# A class containing the necessary tools to clean the raw data and obtain model ready training and testing datasets


# ATRIBUTTES:
# 1. labels, transactions: The original two datasets provided
# 2. data: A dataset merging the two given datasets


# METHODS AND THEIR ARGUMENTS:
# 1. getDataset: Merges the given datasets with a pandas right join (this keeps all transactions with the added labels)

# 2. preprocess: A function which turns the getDataset data into a dataset we can input in our model
#                It adds a new column with a binary label in {0, 1} for non-fraudulent and fraudulent transactions and 
#                the proceeds to label encode the data. Moreover, it gets rid of non essential columns

# 3. getFeatures and getTarget: Splits the dataset into a features dataset for training and a target column (fraud or not)

# 4. getTrainTest: Splits the features and target dataset into model ready training and testing sets
#                  Since the classes are heavily imbalanced (many more non fraudulent transactions than fraudulent), the
#                  function gives the option to use SMOTE oversampling to create synthetic fraudulent transactions. This 
#                  will help during the training phase.

class DataProcessing:

    def __init__(self, transactions= None, labels= None):
        self.labels = pd.read_csv(labels)
        self.transactions = pd.read_csv(transactions)
        self.data = self.getDataset()
    

    def getDataset(self) -> pd.DataFrame:
        return pd.merge(self.labels, self.transactions, on="eventId", how="right")
    

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:

        # Add the fraud flag and drop columns which aren't model-essential
        data["isFraud"] = data["reportedTime"].notna().astype(int)
        model_data = data.drop(columns=["reportedTime", "eventId", "transactionTime"])

        # Label encode the data
        for column in model_data.columns:
            le = LabelEncoder()
            model_data[column] = le.fit_transform(model_data[column])
        
        return model_data
    

    def getFeatures(self, model_data: pd.DataFrame) -> np.array:
        return np.array(model_data.drop("isFraud", axis=1))
    

    def getTarget(self, model_data: pd.DataFrame) -> np.array:
        return model_data.isFraud.values
    

    def getTrainTest(self, features: np.array, target:np.array,  oversampling= False) -> list[np.array]:

        # features = getFeatures(), target = getTarget()

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(features, target, train_size=0.8, random_state=30)

        # If needed, oversample the training sets
        if oversampling:
            oversampling = SMOTE()
            X_resampled, y_resampled = oversampling.fit_resample(X_train, y_train)
            return X_resampled, X_test, y_resampled, y_test

        # Otherwise stick to the regular train and test     
        else:
            return X_train, X_test, y_train, y_test 
        


### class Plotter ###
# A child class of DataProcessing with methods that allow us to visualise the dataset
        
# 1. plotData: Plots the dataset as a scatter plot, with the axes being the proportion of fraudulent 
#              to non fraudulent transactions
        
# 2. comparePlot: Compares the scatter plot of the original dataset with the scatter plot of the dataset after
#                 performing oversampling. This allows to visualise the difference in proportions with and without SMOTE.

class Plotter(DataProcessing):

    def __init__(self, model_data):
        self.model_data = model_data
        self.X = super().getFeatures(self.model_data)
        self.y = super().getTarget(self.model_data)

    def plotData(self) -> None:
        plt.scatter(self.X[self.y == 0, 0], self.X[self.y == 0, 1], label="Non Fraudulent", alpha=0.5, linewidth=0.15)
        plt.scatter(self.X[self.y == 1, 0], self.X[self.y == 1, 1], label="Fraudulent", alpha=0.5, linewidth=0.15, c='r')
        plt.legend()
        plt.show()
    
    def comparePlot(self, X_resampled: np.ndarray, y_resampled: np.ndarray, method: str) -> None:
        plt.subplot(1, 2, 1)
        plt.scatter(self.X[self.y == 0, 0], self.X[self.y == 0, 1], label="Non Fraudulent", alpha=0.5, linewidth=0.15)
        plt.scatter(self.X[self.y == 1, 0], self.X[self.y == 1, 1], label="Fraudulent", alpha=0.5, linewidth=0.15, c='r')
        plt.title('Original Set')

        plt.subplot(1, 2, 2)
        plt.scatter(X_resampled[y_resampled == 0, 0], X_resampled[y_resampled == 0, 1], label="Non Fraudulent", alpha=0.5, linewidth=0.15)
        plt.scatter(X_resampled[y_resampled == 1, 0], X_resampled[y_resampled == 1, 1], label="Fraudulent", alpha=0.5, linewidth=0.15, c='r')
        plt.title(method)
        plt.legend()
        plt.show()