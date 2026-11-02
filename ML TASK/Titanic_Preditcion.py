import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print("Training shape:", train.shape)
print("Testing shape:", test.shape)

print(train.head())
print(test.head())



train = pd.read_csv("train.csv")

print("\nDataset Information:")
print(train.info())

print("\nMissing Values:")
print(train.isnull().sum())

print("\nStatistical Summary:")
print(train.describe())

print("\nSurvival Count:")
print(train["Survived"].value_counts())

print("\nSurvival by Gender:")
print(train.groupby("Sex")["Survived"].mean())

print("\nSurvival by Passenger Class:")
print(train.groupby("Pclass")["Survived"].mean())

sns.countplot(data=train, x="Survived")
plt.title("Titanic Survival Count")
plt.show()

sns.countplot(data=train, x="Sex", hue="Survived")
plt.title("Survival by Gender")
plt.show()

sns.countplot(data=train, x="Pclass", hue="Survived")
plt.title("Survival by Passenger Class")
plt.show()


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Select useful features
features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked"
]

X = train[features].copy()
y = train["Survived"]

X_test = test[features].copy()

age_median = X["Age"].median()

X["Age"] = X["Age"].fillna(age_median)
X_test["Age"] = X_test["Age"].fillna(age_median)

fare_median = X["Fare"].median()

X_test["Fare"] = X_test["Fare"].fillna(fare_median)

embarked_mode = X["Embarked"].mode()[0]

X["Embarked"] = X["Embarked"].fillna(embarked_mode)
X_test["Embarked"] = X_test["Embarked"].fillna(embarked_mode)

X = pd.get_dummies(X, columns=["Sex", "Embarked"], dtype=int)
X_test = pd.get_dummies(X_test, columns=["Sex", "Embarked"], dtype=int)

X_test = X_test.reindex(columns=X.columns, fill_value=0)

print("Preprocessing completed!")
print("\nTraining features:")
print(X.head())

print("\nMissing values:")
print(X.isnull().sum())
print(X_test.isnull().sum())


model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training records:", X_train.shape)
print("Validation records:", X_valid.shape)

model.fit(X_train, y_train)

print("Model training completed!")

y_pred = model.predict(X_valid)

accuracy = accuracy_score(y_valid, y_pred)

print("Model Accuracy:", accuracy * 100)

print(classification_report(y_valid, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_valid, y_pred))

test_predictions = model.predict(X_test)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": test_predictions
})
submission.to_csv("submission.csv", index=False)

print("\nFinal predictions completed!")

print(submission.head())

print("\nPrediction counts:")
print(submission["Survived"].value_counts())

print("\nSubmission file saved successfully!")