import pandas as pd
import pickle
import yaml
import mlflow
import os
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from urllib.parse import urlparse
from dotenv import load_dotenv


def hyper_tuning(X_train, y_train, params_space):
    rfc = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rfc,
                               param_grid=params_space,
                               n_jobs=-1,
                               cv=3,
                               verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search

def train(data_path, model_path):
    data = pd.read_csv(data_path)
    X = data.drop(columns=['Outcome'])
    y = data['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    signature = infer_signature(X_train, y_train)

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
        }
    
    grid_search = hyper_tuning(X_train, y_train, param_grid)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    
    load_dotenv()
    # Load variables saved in .env file
    # Connect to the correct remote server via MLFLOW_TRACKING_URI
    # Authenticate with MLFLOW_TRACKING_USERNAME + MLFLOW_TRACKING_PASSWORD
    # Equal to: mlflow.set_tracking_uri()
    
    mlflow.set_experiment("Diabetes_expt1")
    with mlflow.start_run():
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_param('best_n_estimators', grid_search.best_params_['n_estimators'])
        mlflow.log_param('best_max_depth', grid_search.best_params_['max_depth'])
        mlflow.log_param('best_min_samples_split', grid_search.best_params_['min_samples_split'])
        mlflow.log_param('best_min_samples_leaf', grid_search.best_params_['min_samples_leaf'])
        mlflow.log_text(str(cm), "best_confusion_matrix.txt")
        mlflow.log_text(cr, "best_classification_report.txt")

        if urlparse(mlflow.get_tracking_uri()).scheme != "file":
            mlflow.sklearn.log_model(sk_model=best_model,
                                     artifact_path='artifact_model',
                                     signature=signature,
                                     registered_model_name="Best_model"
                                     )
        else:
            mlflow.sklearn.log_model(sk_model=best_model,
                            artifact_path='artifact_model',
                            signature=signature
                            )

    #save best model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    pickle.dump(best_model, open(model_path, 'wb'))
    print(f"best Accuracy: {accuracy}")

if __name__ == "__main__":
    params = yaml.safe_load(open('params.yaml'))['train']
    train(params['data'], params['model'])