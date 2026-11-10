import pickle
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from credit_fraud_utils_eval import find_best_threshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from credit_fraud_utils_data import(
    load_data ,
    split_feature_target,
    scale_features,
    
    smote_data
)

train_data = load_data("data/train.csv")
val_data = load_data("data/val.csv")

x_train, y_train = split_feature_target(train_data)
x_val, y_val = split_feature_target(val_data)

x_train_fin, y_train_fin = smote_data(x_train, y_train)
model=RandomForestClassifier(random_state=42,class_weight="balanced")
paramater={
    "n_estimators":[5,10,20,30,40,50],
    "max_depth":[3,4,5,6,7,8,9,10],
    "min_samples_split" : [2,5,10],
   "min_samples_leaf" :[1,2,5]
}
grid=GridSearchCV(
    estimator=model,
    param_grid=paramater,
    scoring="f1",
    
    n_jobs=-1
    
)
grid.fit(x_train_fin,y_train_fin)
best_model=grid.best_estimator_
print("Best Parameters:", grid.best_params_)
print("Best CV F1:", grid.best_score_)
best_threshold, best_f1 = find_best_threshold(
   best_model ,
   x_val,
    y_val
)

y_scores = best_model.predict_proba(x_val)[:, 1]


y_pred = (y_scores >= best_threshold).astype(int)

final_f1 = f1_score(y_val, y_pred)

print(classification_report(y_val, y_pred))

print("Balanced Accuracy:", balanced_accuracy_score(y_val, y_pred))

print(confusion_matrix(y_val, y_pred))
print("Final F1 Score:", final_f1)
print("Best Threshold:", best_threshold)




