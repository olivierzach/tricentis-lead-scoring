import pandas as pd

# read in the account data
file_name = "./source/Opportunities_data.csv"
df = pd.read_csv(file_name, engine='python')