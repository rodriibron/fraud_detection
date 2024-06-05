

### PYTHON PACKAGES ###
# Data Handling
import pandas as pd
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Modelling
import xgboost as xgb
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

### DEPENDENCIES FROM OTHER SCRIPTS ###
from data_processing import DataProcessing, Plotter
from xgb_model import xgbModel



# The data file paths
LABELS_FILE = "labels_obf.csv"
TRANSACTIONS_FILE = "transactions_obf.csv"


# The parameter grid we will use to run our model
PARAMETERS = [
    {"eta": 0.2, "gamma": 0.0, "max_depth": 10}, #0
    {"eta": 0.2, "gamma": 0.0, "max_depth": 15}, #1
    {"eta": 0.2, "gamma": 0.1, "max_depth": 14}, #2
    {"eta": 0.2, "gamma": 0.1, "max_depth": 15}, #3
    {"eta": 0.2, "gamma": 0.2, "max_depth": 15}, #4
    {"eta": 0.15, "gamma": 0.2, "max_depth": 17}, #5
    {"eta": 0.1, "gamma": 0.2, "max_depth": 17}, #6
    {"eta": 0.05, "gamma": 0.2, "max_depth": 17}, #7
    {"eta": 0.01, "gamma": 0.2, "max_depth": 17} #8
    
]




dp = DataProcessing(labels= LABELS_FILE, transactions= TRANSACTIONS_FILE)



# A function which returns a processed dataset, ready to be the input of our model
def getData() -> pd.DataFrame:
    dp = DataProcessing(labels= LABELS_FILE, transactions= TRANSACTIONS_FILE)
    return dp.preprocess(dp.data) #model_data



# A function which takes our model ready data and returns a features and target datasets
def modelData(model_data: pd.DataFrame) -> tuple[np.array]:
    dp = DataProcessing(labels= LABELS_FILE, transactions= TRANSACTIONS_FILE)
    X = dp.getFeatures(model_data)
    y = dp.getTarget(model_data)
    return X, y



# A function which runs our model and searches the best combination of hyperparameters
def modelRunner(cm=False, plot_roc=False) -> list[xgb.sklearn.XGBClassifier, float]:

    # Get the data and the training and test datasets
    dp = DataProcessing(labels= LABELS_FILE, transactions= TRANSACTIONS_FILE)
    model_data = getData()
    X, y = modelData(model_data)
    X_train, X_test, y_train, y_test = dp.getTrainTest(features= X, target= y, oversampling= True)

    # Initiate an instance of our model class and run it
    model_runner = xgbModel(X_train, X_test, y_train, y_test, parameters= PARAMETERS)
    runner = model_runner.runModel()

    # These are the dictionaries with the trained models and the statistics for each one of them
    models = runner[0]
    models_stats = runner[1]

    # The model which maximises precision - this way we identify more of the potential fraudulent transactions
    best_model = model_runner.getBestModel(models= models, model_stats= models_stats, metric= "precision")

    # Now let's get the threshold with the maximum number of actual frauds identified
    # But making sure we don't identify over 400 potential frauds
    thresholds = [0.4, 0.5, 0.6, 0.7]
    potential_frauds = {}
    
    for th in thresholds:
        total_potential, total_fraudulent = model_runner.confusionMatrix(model=best_model, show_plot=False, threshold = th) 
        if total_potential > 400:
            continue
        
        else:
            potential_frauds[th] = total_fraudulent
    
    ideal_threshold = max(potential_frauds, key=potential_frauds.get)

    # Let's check how much fraudulent money we could identify
    total_amount = model_runner.totalAmount(model=best_model, model_data=model_data, threshold=ideal_threshold)
    print(f"Total fraud amount identified: {total_amount}")

    # Print plots of the confusion matrix and the ROC
    if cm:
        frauds = model_runner.confusionMatrix(best_model, show_plot=True, threshold= ideal_threshold)
        print(f"Total potential frauds identified: {frauds[0]}")
        print(f"Total actual frauds: {frauds[1]}")
        
    if plot_roc:
        model_runner.plotROC(best_model)
    

    return [best_model, ideal_threshold]







if __name__ == "__main__":
    modelRunner(cm=True, plot_roc=True)







