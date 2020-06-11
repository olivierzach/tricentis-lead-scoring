from data.data_util import *
from sklearn import linear_model
from sklearn.model_selection import train_test_split, cross_validate
import matplotlib.pyplot as plt
import seaborn as sns

# grab the complete data set
pickle_path = './data/pickles/df_model_base.pkl'
df_model_base = pd.read_pickle(pickle_path)

# tuck in id columns
idx_cols = [
    'email',
    'email_domain',
    'company_email_domain',
    'account_id',
    'lead_account_id',
    'passed_to_sales',
    'opportunity_won'
]
df_model_base.set_index(idx_cols, inplace=True)

# columns that were duplicated after join
df_model_base.drop('length_of_path_content_y', axis=1, inplace=True)

# sample data for an unbiased look at the model
df_model = df_model_base.sample(frac=.85)

# split out targets and features - heavily imbalanced labels
model_target_name = 'accepted_by_sales'
model_features = df_model.drop(model_target_name, axis=1)
model_target = df_model[model_target_name].values

# kill columns near zero variance
model_features = variance_threshold(model_features, .025)

# cross validation splits
x_train, x_test, y_train, y_test = train_test_split(
    model_features,
    model_target,
    test_size=.25,
    shuffle=True,
    stratify=model_target,
    random_state=79041284
)

# feature normalization - compute normalization on the training data only
x_train_mean = x_train.mean(axis=0)
x_train_std = x_train.std(axis=0)

# find the columns that are dummy variables - we do not want to scale these
num_cols = [c for c in model_features if not np.isin(model_features[c].dropna().unique(), [-1, 0, 1]).all()]

# scale the test and training set
x_train_scaled, x_test_scaled, fit_train = scale_variables(x_train, x_test, scale_cols=num_cols)

# determine which network features are important to engagement score
plt.rcParams['figure.figsize'] = (12, 8)
df_vimp, df_impt, oob_score = extra_trees_vimp(
    y=df_model[model_target_name],
    df=model_features,
    threshold=0.001,
    plot=True,
    estimators=200,
    depth=200,
    split_sample=.01,
    leaf_sample=.01,
    transform=False
)
plt.show()