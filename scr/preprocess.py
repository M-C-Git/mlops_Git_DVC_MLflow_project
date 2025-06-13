import yaml
import sys
import pandas as pd
import os

params = yaml.safe_load(open('params.yaml'))['preprocess']

def preprocess(input_path, output_path):
    data = pd.read_csv(input_path)

    # interpolate non values with neigbhoring average.
    data = data.interpolate(method='linear')  
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_csv(output_path)

if __name__ == '__main__':
    preprocess(params['input'], params['output'])
    print (f'data preprocess completed and saved in {params["output"]}')