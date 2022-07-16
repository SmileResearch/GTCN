import joblib

path = "trained_models/model_save/Single-Task_2022-05-27-21-40-50_44705_tensor_gcn_tensor_gcn_best_model/roc.joblib"

roc = joblib.load(path)
print(roc)
print("")


