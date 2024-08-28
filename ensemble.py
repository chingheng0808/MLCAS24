import pandas as pd
import os
from predict_latest import predict_result

if not os.path.exists("result_Ensemble"):
    os.makedirs("result_Ensemble")

EPOCH = 800

df_test = pd.read_csv(
    "MLCAS24_Competition_data/test/Test/GroundTruth/test_HIPS_HYBRIDS_2023_V2.3.csv"
)

models_list = os.listdir('models')
for model in models_list:
    if f'epoch-{EPOCH}' in model:
        predict_result(f'models/{model}')

result_list = os.listdir('results_test')
df_list = []

for result in result_list:
    df = pd.read_csv(f'results_test/{result}')
    df_list.append(df)

for idx, row in df_test.iterrows():
    mean_yield = 0.
    for df in df_list:
        mean_yield += float(df.at[idx, "yieldPerAcre"])

    df_test.at[idx, "yieldPerAcre"] = float(mean_yield/len(df_list))

df_test.to_csv(f'result_Ensemble/result_test_ensemble{EPOCH}.csv', index=False)
