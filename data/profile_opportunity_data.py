import pandas as pd

# read in the account data
file_name = "./source/Opportunities_data.csv"
df = pd.read_csv(file_name, engine='python')

# TODO: profile data
# TODO: clean data - missing data, categorical data
# TODO: find external keys
# TODO: build key modeling features
# TODO: think about time - when do we get this data?