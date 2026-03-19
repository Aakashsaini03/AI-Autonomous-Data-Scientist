from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import pandas as pd

def train_model(df, target):
    X = df.drop(columns=[target])
    y = df[target]

    # Encode categorical variables
    X = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    if y.dtype == 'object':
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = accuracy_score(y_test, preds)
        metric = "Accuracy"
    else:
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = mean_squared_error(y_test, preds)
        metric = "MSE"

    return model, score, metric
