import pandas as pd
import numpy as np
df = pd.read_csv('~/code/bo/fin/suggestions_50_fin.csv')
print(df['expected_improvement'].median())
