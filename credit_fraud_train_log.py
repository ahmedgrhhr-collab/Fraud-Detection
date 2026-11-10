import pickle
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from credit_fraud_utils_eval import find_best_threshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
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
x_train_scale,x_val_sacle,scale =scale_features(x_train,x_val)
x_train_fin, y_train_fin = smote_data(x_train_scale, y_train)
model=LogisticRegression( random_state=42)
model.fit(x_train_fin,y_train_fin)

best_threshold, best_f1 = find_best_threshold(
   model ,
   x_val_sacle,
    y_val
)

y_scores = model.predict_proba(x_val_sacle)[:, 1]


y_pred = (y_scores >= best_threshold).astype(int)

final_f1 = f1_score(y_val, y_pred)

print(classification_report(y_val, y_pred))

print("Balanced Accuracy:", balanced_accuracy_score(y_val, y_pred))

print(confusion_matrix(y_val, y_pred))
print("Final F1 Score:", final_f1)
print("Best Threshold:", best_threshold)
