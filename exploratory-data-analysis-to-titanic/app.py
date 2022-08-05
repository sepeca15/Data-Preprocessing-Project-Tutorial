import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, mean_squared_error
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb

df_raw = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/random-forest-project-tutorial/main/titanic_train.csv',index_col=0)
df_raw = df_raw.drop(columns='Cabin', axis=1)
df_raw['Age'] = df_raw['Age'].fillna(df_raw['Age'].mean())
df_raw['Embarked'] = df_raw['Embarked'].fillna(df_raw['Embarked'].mode()[0])
df = df_raw.copy()
X = df.drop(columns=['Ticket', 'Name', 'Survived'])
y= df['Survived']
X [['Sex', 'Embarked']] = X [['Sex', 'Embarked']].astype('category')
X['Sex'] = X['Sex'].cat.codes
X['Embarked'] = X['Embarked'].cat.codes
X_train, X_test, y_train, y_test = train_test_split (X, y, random_state=15)
clf_GB = GradientBoostingClassifier(n_estimators=100)
clf_GB.fit(X_train, y_train)
xgb_model = xgb.XGBRegressor(objective="reg:linear", random_state=42)
xgb_model.fit(X_train, y_train)
filename = '/workspace/Data-Preprocessing-Project-Tutorial/models/model-clg.GB.pickle'
pickle.dump(clf_GB, open(filename, 'wb'))
filename1 = '/workspace/Data-Preprocessing-Project-Tutorial/models/model-xgb-reg.pickle'
pickle.dump(xgb_model, open(filename1, 'wb'))