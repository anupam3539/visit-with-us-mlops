import json
from pathlib import Path
import joblib, mlflow, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[2]
DEPLOY = ROOT / "tourism_project" / "deployment"

def main():
    X_train, X_test = pd.read_csv(ROOT/"Xtrain.csv"), pd.read_csv(ROOT/"Xtest.csv")
    y_train = pd.read_csv(ROOT/"ytrain.csv")["ProdTaken"]
    y_test = pd.read_csv(ROOT/"ytest.csv")["ProdTaken"]
    categorical = X_train.select_dtypes(include="object").columns.tolist()
    numeric = X_train.select_dtypes(exclude="object").columns.tolist()
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    pipe = Pipeline([("preprocessor", preprocessor),
                     ("classifier", RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=1))])
    grid = {
        "classifier__n_estimators": [150, 250],
        "classifier__max_depth": [8, 12, None],
        "classifier__min_samples_leaf": [1, 3],
        "classifier__max_features": ["sqrt"],
    }
    search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=3, n_jobs=1, return_train_score=True)
    mlflow.set_tracking_uri((ROOT/"mlruns").as_uri())
    mlflow.set_experiment("wellness-tourism-purchase")
    with mlflow.start_run(run_name="random-forest-grid-search"):
        search.fit(X_train, y_train)
        pred, probability = search.predict(X_test), search.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probability),
            "best_cv_roc_auc": search.best_score_,
        }
        mlflow.log_params(search.best_params_)
        mlflow.log_metrics(metrics)
        DEPLOY.mkdir(parents=True, exist_ok=True)
        joblib.dump(search.best_estimator_, DEPLOY/"tourism_model.joblib")
        pd.DataFrame(search.cv_results_).to_csv(ROOT/"tuning_results.csv", index=False)
        (ROOT/"best_parameters.json").write_text(json.dumps(search.best_params_, indent=2))
        (ROOT/"model_metrics.json").write_text(json.dumps(metrics, indent=2))
        mlflow.log_artifact(str(ROOT/"tuning_results.csv"))
    print("Best parameters:", search.best_params_)
    print("Metrics:", {k: round(v, 4) for k, v in metrics.items()})
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))
    print("Classification report:\n", classification_report(y_test, pred, digits=3, zero_division=0))
    print("Saved model:", DEPLOY/"tourism_model.joblib")

if __name__ == "__main__":
    main()
