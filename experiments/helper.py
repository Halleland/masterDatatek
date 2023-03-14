from datetime import datetime

import pandas as pd

import sys

sys.path.append('C:\\Users\\Martin\\Documents\\GitHub\\master\\evaluation')

import trainer

def load_all_data(paths):
    data = pd.DataFrame()
    for path in paths:
        df = pd.read_table(path)
        data = data.append(df, ignore_index=True)
    return data



def run_experiment(trainer, output_folder, name, num_times=3):
    exp_start = datetime.now().strftime('%d%m%Y_%H_%M_%S')
    print(f'start:{exp_start}')
    outputs = []
    results = []
    for i in range(num_times):
        out = trainer.train(trainer.params)
        res = trainer.evaluate(out)
        results.append(res)
        outputs.append({'topics':out})
    df_res = pd.DataFrame.from_records(results)
    df_out = pd.DataFrame(outputs)
    
    df_res.to_csv(f'{output_folder}/{name}_metrics_{exp_start}.csv')
    df_out.to_csv(f'{output_folder}/{name}_topics_{exp_start}.csv')

    exp_end = datetime.now().strftime('%d%m%Y_%H_%M_%S')
    print(f'end:{exp_start}')