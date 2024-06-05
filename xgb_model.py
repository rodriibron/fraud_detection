
import pandas as pd 
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay




### class xgbModel ###
# A class which contains all necessary tools to run our XGB Classifier


# ATRIBUTTES:
# 1. X_train, y_train, X_test, y_test: The training and test datasets we will use for our model. These have been obtained
#    by making use of the DataProcessing class, in order to clean the original data and split it for train and test

# 2. parameters: A grid of hyperparameters we will test. We will pick the hyperparameters which maximises the amount of 
#    potentially fraudulent transactions identified


# METHODS AND THEIR ARGUMENTS:
# 1. runModel: Our model runner. Step by step comments are added in the function itself
#              It returns two dictionaries, both with the same keys (the run name): one of them contains the 
#              XGBClassifier objects which are the trained models, and the other contains pd.DataFrames with model stats
    

# 2. getBestModel: Runs through the statistics of each one of the models and gives back the one which performed the better.
#                  Returns a XGBClassifier object. Only parameter is the aforementioned model and model stats dictionaries


# 3. confusionMatrix: This returns a confusion matrix for visualisation of our model's predictions.
#                     Our XBGC model classifies transactions as fraudulent (1) or non-fraudulent (0) based in their
#                     probabability outputs (>0.5 is fraudulent).

#                     However, confusionMatrix lets you change this probability threshold to a different number, and returns
#                     another confusion matrix which allows you to check if more fraudulent transactions would have been
#                     identified with a different threshold (e.g. >0.4 is fraudulent)


# 4. totalAmount: Gives you the total amount of fraudulent money that the model has correctly identified for a given
#                 threshold - in other words, for a given classification threshold, this function returns how much
#                 fraudulent money we have stopped from being moved.
#                 The function takes as arguments: a model (xgb.sklearn.XGBClassifier object) which would usually be our
#                 best model, the model data (pd.DataFrame object) and our chose threshold (float)

# 5. plotROC: Plots the ROC curve of a given model. The only argument it takes is the model we want to plot (which would            
#             also be a xgb.sklearn.XGBClassifier object).


class xgbModel:

    def __init__(self, X_train, X_test, y_train, y_test, parameters):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.parameters = parameters
        #self.models = self.runModel()


    def runModel(self) -> list[dict]:

        models = {}
        model_stats = {}

        # Run through the parameter grid
        for i, params in enumerate(self.parameters):
            run_name = f"fraud-model-{i}"
            print(run_name)
            model = xgb.XGBClassifier(scale_pos_weight=100,
                                      objective="reg:logistic",
                                      max_depth=params["max_depth"],
                                      gamma=params["gamma"],
                                      eta=params["eta"],
                                      use_label_encoder=False,
            )
            X_train = self.X_train
            y_train = self.y_train

            # Fit and save the model
            model.fit(X_train, y_train)
            models[run_name] = model
            

            y_iterate_pred = model.predict(self.X_test)

            # Save the model metrics as a dictionary
            # Keys are the models, values are the dataframes with the statistics
            acc = accuracy_score(self.y_test, y_iterate_pred)
            rec = recall_score(self.y_test, y_iterate_pred)
            pre = precision_score(self.y_test, y_iterate_pred)
            m_eval = pd.DataFrame({"accuracy":[acc], "recall":[rec], "precision":[pre]})
            model_stats[model] = m_eval
        
        return [models, model_stats]
    


    @staticmethod
    def getBestModel(models: dict, model_stats: dict, metric: str) -> xgb.sklearn.XGBClassifier:

        stats = []
        for key in model_stats.keys():
            df = model_stats[key]
            stats.append(df.loc[0, f"{metric}"])
        
        max_idx = stats.index(max(stats))
        print(f"Max {metric} {max(stats)} found at {max_idx}")
        return models[f"fraud-model-{max_idx}"]
    


    def confusionMatrix(self, model: xgb.sklearn.XGBClassifier, show_plot=False, threshold = None) -> tuple[int]:

        # The predicted probabilities
        y_pred_prob = model.predict_proba(self.X_test)[:,1]

        # If we want to change threshold, use the probabilities
        if threshold:
            threshold_change = lambda x: 1 if x>threshold else 0
            y_th = []
            for prob in y_pred_prob:
                y_th.append(threshold_change(prob))

            cm = confusion_matrix(self.y_test, y_th, labels=model.classes_)     

            # Show the confusion matrix of the model with the new threshold   
            if show_plot:
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
                disp.plot()
                plt.show()

            # Calculate total potential frauds and total actual frauds          
            total_potential_fraud = cm[0][1] + cm[1][1]
            total_fraud = cm[1][1]
            #print(f"Total Potential Fraudulent Transactions {total_potential_fraud}")
            #print(f"Total Actual Fraudulent Transactions {total_fraud}")
            return total_potential_fraud, total_fraud
        
        else:
            y_pred = y_pred_prob.round().astype(int)
            
            cm = confusion_matrix(self.y_test, y_pred, labels=model.classes_)
            if show_plot:
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
                disp.plot()
                plt.show()

            total_potential_fraud = cm[0][1] + cm[1][1]
            total_fraud = cm[1][1]
            #print(f"Total Potential Fraudulent Transactions {total_potential_fraud}")
            #print(f"Total Actual Fraudulent Transactions {total_fraud}")
            return total_potential_fraud, total_fraud
        
    
    def totalAmount(self, model: xgb.sklearn.XGBClassifier, model_data: pd.DataFrame, threshold:float) -> int:

        # Get the probabilities and define the threshold change function
        y_prob = model.predict_proba(self.X_test)[:,1]
        threshold_change = lambda x: 1 if x>threshold else 0
        
        # Change the predictions of y_prob based on the threshold
        y_th = []
        for prob in y_prob:
            y_th.append(threshold_change(prob))
        
        # Create a dataframe with the model original data and the new predictions
        y_th = pd.Series(y_th)
        model_data["pred"] = y_th

        # Subset the to the rows of the original data which coincide with the y_prob rows and subset the fraudulent
        # Then return the sum of amounts in the transactions
        predictions = model_data[model_data["pred"].notna()]
        fraudulent_transactions = predictions[predictions["pred"] == 1]


        return fraudulent_transactions["transactionAmount"].sum()
    

    def plotROC(self, model: xgb.sklearn.XGBClassifier) -> None:

        # Extract the probabilities, the false positive and false negative rates, and their ROC-AUC score
        y_prob = model.predict_proba(self.X_test)[:,1]
        fpr, tpr, _ = roc_curve(self.y_test, y_prob)
        roc_auc = roc_auc_score(self.y_test, y_prob)
        
        # Plot the ROC
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()
