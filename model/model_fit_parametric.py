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

# lasso to reduce dimension
lasso_c = linear_model.SGDClassifier(
    loss='log',
    penalty='l1',
    alpha=.05,
    fit_intercept=True,
    shuffle=True,
    class_weight={0: 1, 1: 100},
    verbose=10
)
# fit and extract
lasso_fit = lasso_c.fit(x_train_scaled, y_train)
lasso_weights = lasso_fit.coef_
print(f"MODEL SCORE: {lasso_fit.score(x_train_scaled, y_train)}")
print(f"COEFFICIENT COUNT: {len(np.nonzero(lasso_weights)[1])}")

# cross validate the lasso
cross_validate(
    lasso_fit,
    X=x_train_scaled,
    y=y_train,
    cv=5,
    scoring=(
        'accuracy',
        'balanced_accuracy',
        'precision',
        'recall',
        'roc_auc'
    )
)

# look at coefficients - how many are left after regularization?
print(f"Remaining variables: {np.sum(abs(lasso_weights) > 0)}")
# subset original frame to see what the columns are
non_zero_idx = list(np.nonzero(lasso_weights)[0])
best_features_list = df_model_features.iloc[:, non_zero_idx].columns
df_best_features = df_model_features[best_features_list]
df_best_features.mean(axis=0)
# subset features and re-fit
x_train_best = x_train[best_features_list]
x_test_best = x_test[best_features_list]
# fit a vanilla least squares model on the best coefficients
least_squares = linear_model.LinearRegression(
    normalize=True,
    fit_intercept=True
)
# fit and extract
least_squares_fit = least_squares.fit(x_train_best, y_train)
least_squares_weights = lasso_fit.coef_
least_squares_fit.score(x_train_best, y_train)
# cross validate the regression
cross_validate(
    least_squares_fit,
    X=x_train_best,
    y=y_train.values.ravel(),
    cv=10,
    scoring=(
        'neg_mean_absolute_error',
        'neg_mean_squared_error',
        'neg_median_absolute_error'
    )
)
# build a coefficient dictionary
ls_dict = {}
for idx, v in enumerate(least_squares_weights):
    if abs(v) >= 0:
        name = df_model_features.iloc[:, idx].name
        ls_dict[name] = v
print(ls_dict)
# convert dict to frame for plotting
ls_df = pd.DataFrame.from_dict(ls_dict, orient='index').reset_index()
ls_df.columns = ['variable', 'beta']
ls_df['variable'] = ls_df['variable'].str.replace('DMA_', '')
plt.rcParams['figure.figsize'] = (12, 8)
sns.barplot(
    y='variable',
    x='beta',
    data=ls_df[abs(ls_df['beta']) >= 0.0001].sort_values('beta', ascending=False),
    palette='Blues_r'
).set_title('Network Engagement Affect on GD Engagement')
plt.axvline(np.median(ls_df['beta']), linestyle='--', color='orange')
plt.ylabel('')
plt.show()
# model multiple feature groups instead of all together
feature_groups = [
    'VIEWS_mean',
    'ENGAGEMENT_SCORE_mean',
    'MINS_mean',
    'VIEWS_sum',
    'ENGAGEMENT_SCORE_sum',
    'MINS_sum',
    'VIEWS_count',
    'ENGAGEMENT_SCORE_count',
    'MINS_count',
]
for i in feature_groups:
    # subset to only metric and agg filters
    filter_cols = [c for c in df_full.columns if i in c]
    print(filter_cols)
    # only want to model with network engagement
    df_model_features = df_full[filter_cols + ['ENGAGEMENT_SCORE']].drop([model_target], axis=1)
    # cross validation splits
    x_train, x_test, y_train, y_test = train_test_split(
        df_model_features,
        df_model_target,
        test_size=.35,
        shuffle=True,
        random_state=65244
    )
    # scale variables to account for variable magnitudes across shows
    x_train_scaled, x_test_scaled, _ = scale_variables(
        train=x_train,
        test=x_test,
        scale_cols=x_train.columns
    )
    # lasso to reduce dimension
    lasso_c = linear_model.Lasso(
        alpha=.00001,
        random_state=48167,
        max_iter=5000,
        fit_intercept=True,
        tol=1e-6,
        normalize=True,
        positive=True
    )
    # fit and extract
    lasso_fit = lasso_c.fit(x_train_scaled, np.array(y_train).ravel())
    lasso_weights = lasso_fit.coef_
    print(f"Model Score: {lasso_fit.score(x_train_scaled, y_train)}")
    # cross validate the lasso
    cross_validate(
        lasso_fit,
        X=x_train_scaled,
        y=y_train.values.ravel(),
        cv=10,
        scoring=(
            'neg_mean_absolute_error',
            'neg_mean_squared_error',
            'neg_median_absolute_error'
        )
    )
    # look at coefficients - how many are left after regularization?
    print(f"Remaining variables: {np.sum(abs(lasso_weights) > 0)}")
    # subset original frame to see what the columns are
    non_zero_idx = list(np.nonzero(lasso_weights)[0])
    best_features_list = df_model_features.iloc[:, non_zero_idx].columns
    df_best_features = df_model_features[best_features_list]
    df_best_features.mean(axis=0)
    # subset features and re-fit
    x_train_best = x_train[best_features_list]
    x_test_best = x_test[best_features_list]
    # fit a vanilla least squares model on the best coefficients
    least_squares = linear_model.LinearRegression(
        normalize=True,
        fit_intercept=True
    )
    # fit and extract
    least_squares_fit = least_squares.fit(x_train_best, y_train)
    least_squares_weights = lasso_fit.coef_
    least_squares_fit.score(x_train_best, y_train)
    # cross validate the regression
    cross_validate(
        least_squares_fit,
        X=x_train_best,
        y=y_train.values.ravel(),
        cv=10,
        scoring=(
            'neg_mean_absolute_error',
            'neg_mean_squared_error',
            'neg_median_absolute_error'
        )
    )
    # build a coefficient dictionary
    ls_dict = {}
    for idx, v in enumerate(least_squares_weights):
        if abs(v) >= 0:
            name = df_model_features.iloc[:, idx].name
            ls_dict[name] = v
    print(ls_dict)
    # convert dict to frame for plotting
    ls_df = pd.DataFrame.from_dict(ls_dict, orient='index').reset_index()
    ls_df.columns = ['variable', 'beta']
    ls_df['variable'] = ls_df['variable'].str.replace('DMA_', '')
    plt.rcParams['figure.figsize'] = (12, 8)
    sns.barplot(
        y='variable',
        x='beta',
        data=ls_df[abs(ls_df['beta']) >= 0.0001].sort_values('beta', ascending=False),
        palette='Blues_r'
    ).set_title('Network Engagement Affect on GD Engagement')
    plt.axvline(np.median(ls_df['beta']), linestyle='--', color='orange')
    plt.ylabel('')
    plt.show()