import joblib
import pandas as pd
from preprocess import preprocess_loan_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

train_df = pd.read_csv('data/train_u6lujuX_CVtuZ9i.csv')
test_df = pd.read_csv('data/test_Y3wMUE5_7gLdaTN.csv')
train_processed, test_processed, preproc = preprocess_loan_data(train_df, test_df)
X = train_processed.drop('Loan_Status', axis=1)
y = train_processed['Loan_Status']
X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

model = joblib.load('models/random_forest.pkl')
X_val_sel = X_val[model.feature_names_in_]
pred = model.predict(X_val_sel)
acc = accuracy_score(y_val,pred)
print('Validation accuracy (saved model):', acc)
print('Feature count:', len(model.feature_names_in_))
print('Model params sample:', {k:model.get_params().get(k) for k in ['n_estimators','max_depth','min_samples_split','min_samples_leaf']})
