import pandas as pd

import sklearn

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV

# ーここからtrain読み込みーーーー
path = ("/workspaces/Titanic/train.csv")

data = pd.read_csv(path)

data['Age'].fillna(data['Age'].mean(), inplace=True)

data['SibSp'].fillna(data['SibSp'].mean(), inplace=True)

data['Pclass'].fillna(data['Pclass'].mean(), inplace=True)

data['Parch'].fillna(data["Parch"].mean(), inplace=True)

data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

data["Fare"].fillna(data["Fare"].mean(), inplace=True)

data["Embarked"].fillna("A", inplace=True)

# 搭乗地不明はA

data = pd.get_dummies(data,columns = ["Embarked"], drop_first=False)

data = pd.get_dummies(data,columns = ["Sex"], drop_first=True)

feature = ["Pclass","Age","FamilySize","Sex_male","Fare","Embarked_S","Embarked_C","Embarked_Q"]

x = data[feature]

y = data.Survived

#ーここから育成・パラメータ絞り込みーーーー
param_grid = {'max_depth': [2,3,4,5,7,9],
              'min_samples_split': [1,5,9,13,17,21]}
grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid)

grid_search.fit(x, y)

best_param = grid_search.best_estimator_

print("Best parameters:", grid_search.best_params_)
# ーここからtrain読み込みーーーー
test_path = ("/workspaces/Titanic/test.csv")

test_data = pd.read_csv(test_path)

test_data['Age'].fillna(test_data['Age'].mean(), inplace=True)

test_data['SibSp'].fillna(test_data['SibSp'].mean(), inplace=True)

test_data['Pclass'].fillna(test_data['Pclass'].mean(), inplace=True)

test_data['Parch'].fillna(test_data["Parch"].mean(), inplace=True)

test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1

test_data["Fare"].fillna(test_data["Fare"].mean(), inplace=True)

test_data["Embarked"].fillna("A", inplace=True)

# 搭乗地不明はA

test_data = pd.get_dummies(test_data,columns = ["Embarked"], drop_first=False)

test_data = pd.get_dummies(test_data,columns = ["Sex"], drop_first=True)

test_x = test_data[feature]

y_predictions = best_param.predict(test_x)

outnum = test_data["PassengerId"]

output = pd.DataFrame({"PassengerId": outnum, "Survived": y_predictions})

output.to_csv("/workspaces/Titanic/output.csv", index=False)