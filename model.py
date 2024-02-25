import joblib 
import pandas as pd
import numpy as np
import os

# load the model file
curr_path = os.path.dirname(os.path.realpath(__file__))
xgb_model = joblib.load(curr_path + "/model/wbb_xgb_model2.joblib")
gbr_model = joblib.load(curr_path + "/model/wbb_gbr_model2.joblib")
clustergin_model = joblib.load(curr_path + "/model/clustering.joblib")
scaler = joblib.load(curr_path + "/model/scaler.joblib")



# function to predict the yield
def predict_yield(attributes: np.ndarray):
    X_clus = attributes[:, -4:]
    new_data_scaled = scaler.transform(X_clus)

    cluster_predictions = clustergin_model.predict(new_data_scaled)
    cluster_predictions_reshaped = cluster_predictions.reshape(-1, 1)
    new_data_for_prediction = np.concatenate([attributes, cluster_predictions_reshaped], axis=1)
    # new_data_scaled.append(cluster_predictions)
    
    pred = gbr_model.predict(new_data_for_prediction)
    print("Yield predicted")

    return pred[0]