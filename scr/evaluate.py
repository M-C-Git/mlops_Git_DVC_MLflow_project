import pandas as pd
from sklearn.metrics import accuracy_score
import mlflow
from dotenv import load_dotenv
import yaml
import pickle

def evaluate(data_path, model_path):
    data = pd.read_csv(data_path)
    X = data.drop(columns=['Outcome'])
    y = data['Outcome']
    model = pickle.load(open(model_path, 'rb'))
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    load_dotenv('scr/.env')
    mlflow.log_metric('eval_accuracy', accuracy)
    print (f'the evaluation accuracy is {accuracy}')


if __name__ == "__main__":
    params = yaml.safe_load(open('params.yaml'))['evaluate']
    evaluate(params['data'], params['model'])
