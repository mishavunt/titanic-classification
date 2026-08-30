import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import kagglehub
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import os

    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import OrdinalEncoder
    return (
        GradientBoostingClassifier,
        GridSearchCV,
        LogisticRegression,
        OrdinalEncoder,
        RandomForestClassifier,
        StandardScaler,
        cross_val_score,
        kagglehub,
        np,
        os,
        pd,
        px,
    )


@app.cell
def _(kagglehub, os, pd):
    folder_path = kagglehub.dataset_download("dimplebathija/titanic-machine-learning-from-disaster")

    file_train_path = os.path.join(folder_path, 'Titanic_train.csv')
    file_test_path = os.path.join(folder_path, 'Titanic_test.csv')

    train_start = pd.read_csv(file_train_path)
    X_test_start = pd.read_csv(file_test_path)
    return X_test_start, train_start


@app.cell
def _(train_start):
    train_start.isnull().sum().sort_values(ascending=False)
    return


@app.cell
def _(X_test_start):
    X_test_start.isnull().sum().sort_values(ascending=False)
    return


@app.cell
def _(OrdinalEncoder, train_start):
    train = train_start.drop(columns=["PassengerId", "Name", 'Ticket'])

    train["Cabin"] = train["Cabin"].str[0]

    ordinal_enc = OrdinalEncoder()

    columns_to_encode = ["Sex", "Cabin", "Embarked"]
    train[columns_to_encode] = ordinal_enc.fit_transform(train[columns_to_encode])
    train['Age'] = train['Age'].fillna(train['Age'].median())
    train['Cabin'] = train['Cabin'].fillna(-1)
    return columns_to_encode, ordinal_enc, train


@app.cell
def _(X_test_start, columns_to_encode, ordinal_enc, train):
    X_test = X_test_start.drop(columns=["PassengerId", "Name", 'Ticket'])
    X_test["Cabin"] = X_test["Cabin"].str[0]
    X_test[columns_to_encode] = ordinal_enc.transform(X_test[columns_to_encode])
    X_test['Age'] = X_test['Age'].fillna(train['Age'].median())
    X_test['Cabin'] = X_test['Cabin'].fillna(-1)
    X_test['Fare'] = X_test['Fare'].fillna(train['Fare'].median())
    return (X_test,)


@app.cell
def _(train):
    X_train = train.drop(columns='Survived')
    y_train = train["Survived"]
    return X_train, y_train


@app.cell
def _(StandardScaler, pd, train):
    scaled_for_selection_types = StandardScaler().fit_transform(train)
    X_scaled_for_types = pd.DataFrame(scaled_for_selection_types, columns=train.columns, index=train.index)
    X_scaled_for_types
    return (X_scaled_for_types,)


@app.cell
def _(X_scaled_for_types):
    corr = X_scaled_for_types.corr()["Survived"]
    # так
    corr = corr.abs()
    # или так
    # corr.loc[corr < 0] = corr * (-1)
    corr = corr.sort_values(ascending=False)
    corr
    return


@app.cell
def _(RandomForestClassifier, X_scaled_for_types, pd, y_train):
    clf_for_types = RandomForestClassifier()
    clf_for_types.fit(X_scaled_for_types, y_train)

    importance = pd.Series(clf_for_types.feature_importances_,  index=X_scaled_for_types.columns)
    importance = importance.drop(index='Survived')
    importance.sort_values(ascending=False)
    return (importance,)


@app.cell
def _(importance, px):
    px.bar(importance)
    return


@app.cell
def _(X_test, X_train):
    X_train["Is_alone"] = ((X_train["Parch"] + X_train["SibSp"]) == 0).astype(int)
    X_test["Is_alone"] = ((X_test["Parch"] + X_test["SibSp"]) == 0).astype(int)
    return


@app.cell
def _(X_test, X_train):
    X_train_new = X_train.drop(columns=["Embarked", "Parch", "SibSp"])
    X_test_new = X_test.drop(columns=["Embarked", 'Parch', 'SibSp'])
    return X_test_new, X_train_new


@app.cell
def _(StandardScaler, X_test_new, X_train_new):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train_new)
    X_test_scaled = scaler.transform(X_test_new)
    return X_test_scaled, X_train_scaled


@app.cell
def _(X_test_scaled, X_train_scaled, np):
    print(np.isnan(X_train_scaled).sum(), np.isnan(X_test_scaled).sum())
    return


@app.cell
def _(X_test_scaled, X_train_scaled, np):
    print(np.isinf(X_train_scaled).sum(), np.isinf(X_test_scaled).sum())
    return


@app.cell
def _(
    GradientBoostingClassifier,
    LogisticRegression,
    RandomForestClassifier,
    X_train_scaled,
    cross_val_score,
    np,
    y_train,
):
    model={
        'Random Forest': RandomForestClassifier(),
        'Logistic Regerssion': LogisticRegression(max_iter=1000, solver="liblinear"),
        'Gradient Boosting Classifier': GradientBoostingClassifier()
    }
    for i in range(0,4):
        print(i+1)
        for name, models in model.items():
            score = cross_val_score(models, X_train_scaled, y_train, cv=5)
            print(f"{name}: {np.mean(score):.3f}")
    return


@app.cell
def _(GridSearchCV, LogisticRegression, X_train_scaled, y_train):
    lr_params={
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l1','l2'],
        'solver': ["liblinear"]
    }

    lrc = LogisticRegression()
    grid_lr = GridSearchCV(lrc, param_grid=lr_params, cv=5)
    grid_lr.fit(X_train_scaled, y_train)
    best_lr = grid_lr.best_estimator_
    grid_lr.best_score_
    return


@app.cell
def _(GridSearchCV, RandomForestClassifier, X_train_scaled, y_train):
    rf_params={
        'n_estimators': [100, 200, 300],
        'min_samples_split': [2, 5],
        "max_depth": [None, 5, 7]
    }

    rfc = RandomForestClassifier()
    grid_rb = GridSearchCV(rfc, param_grid=rf_params, cv=5)
    grid_rb.fit(X_train_scaled, y_train)
    grid_rb.best_score_
    return (grid_rb,)


@app.cell
def _(GradientBoostingClassifier, GridSearchCV, X_train_scaled, y_train):
    gb_params={
        "n_estimators": [100, 175],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3,5,7]
    }

    gbc = GradientBoostingClassifier()
    grid_gb = GridSearchCV(gbc, param_grid=gb_params, cv=5)
    grid_gb.fit(X_train_scaled, y_train)
    grid_gb.best_score_
    return (grid_gb,)


@app.cell
def _(grid_rb):
    grid_rb.best_params_
    return


@app.cell
def _(grid_gb):
    grid_gb.best_params_
    return


@app.cell
def _(GradientBoostingClassifier, X_train_scaled, y_train):
    best_clf = GradientBoostingClassifier()

    best_clf.fit(X_train_scaled, y_train)
    return (best_clf,)


@app.cell
def _(X_test_scaled, X_test_start, best_clf, os, pd):
    y_pred = best_clf.predict(X_test_scaled)

    submission = pd.DataFrame({
        "PassengerId": X_test_start["PassengerId"],
        "Survived": y_pred
    })

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "submission.csv")
    submission.to_csv(output_path, index=False)
    return


if __name__ == "__main__":
    app.run()
