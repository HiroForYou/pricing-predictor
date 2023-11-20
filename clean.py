import pandas as pd

from utils import complete_missing_rows

input_file = './data/2019-2023.csv'
output_file = './data/2019-2023_without_facturacion.csv'

df = pd.read_csv(input_file)

df = df.drop('Facturación', axis=1)
df.to_csv(output_file, index=False)