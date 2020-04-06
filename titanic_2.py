# %%
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

train = pd.read_csv("/Users/chad/visualprojects/kaggle/titiantic/train.csv")
test = pd.read_csv("/Users/chad/visualprojects/kaggle/titiantic/test.csv")

def deal_age(df):
    df["Age"] = df["Age"].fillna(-0.5)      # try -0.5 -> 0
    cut_points = [-1, 0, 5, 12, 18, 35, 60, 100]
    label_names = ["Missing","Infant","Child","Teenager","Young Adult","Adult","Senior"]
    df['Age_categories'] = pd.cut(df['Age'], cut_points, labels=label_names)
    return df

def build_dummies(df, column_name):
    dummies = pd.get_dummies(df[column_name], prefix=column_name)
    df = pd.concat([df, dummies], axis=1)
    return df

# %%
train = deal_age(train)
test = deal_age(test)

for col in ["Age_categories", "Pclass", "Sex"]:
    train = build_dummies(train, col)
    test = build_dummies(test, col)

print(train.columns)


# %%
columns = ['SibSp','Parch','Fare','Cabin','Embarked']
print(train[columns].describe(include='all', percentiles=[]))

# %%
from sklearn.preprocessing import minmax_scale
test["Fare"] = test["Fare"].fillna(train["Fare"].mean())
columns = ["SibSp","Parch","Fare"]

train["Embarked"] = train["Embarked"].fillna("S")
test["Embarked"] = test["Embarked"].fillna("S")

train = build_dummies(train, "Embarked")
test = build_dummies(test, "Embarked")

for col in columns:
    train[col+ "_scaled"] = minmax_scale(train[col])
    test[col+ "_scaled"] = minmax_scale(test[col])

columns = ['Age_categories_Missing', 'Age_categories_Infant',
       'Age_categories_Child', 'Age_categories_Teenager',
       'Age_categories_Young Adult', 'Age_categories_Adult',
       'Age_categories_Senior', 'Pclass_1', 'Pclass_2', 'Pclass_3',
       'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S',
       'SibSp_scaled', 'Parch_scaled', 'Fare_scaled']

lr = LogisticRegression()
lr.fit(train[columns], train["Survived"])

coefficients = lr.coef_ ## y = ax + b for a
intercept = lr.intercept_ ## y = ax + b for b
print(coefficients)
print(intercept)

feature_importannce = pd.Series(coefficients[0], index=train[columns].columns)
ordered_feature_importance = feature_importannce.abs().sort_values()
ordered_feature_importance.plot.barh()
plt.show()

# %%
from sklearn.model_selection import cross_val_score

columns = ['Age_categories_Infant', 'SibSp_scaled', 'Sex_female', 'Sex_male',
       'Pclass_1', 'Pclass_3', 'Age_categories_Senior', 'Parch_scaled']
all_x = train[columns]
all_y = train["Survived"]

lr = LogisticRegression()
scores = cross_val_score(lr, all_x, all_y, cv=10)
accuracy = scores.mean()
print(accuracy)

# %%
# lr.fit(all_x, all_y)
# test_predicitions = lr.predict(test[columns])

# test_ids = test["PassengerId"]
# submission_df = {"PassengerId": test_ids, "Survived": test_predicitions}
# submission = pd.DataFrame(submission_df)
# submission.to_csv("submission1.csv", index=False)

titles = {
    "Mme":         "Mrs",
    "Ms":          "Mrs",
    "Mrs" :        "Mrs",
    "Countess":    "Royalty",
    "Lady" :       "Royalty"
}

extracted_titles = train["Name"].str.extract()