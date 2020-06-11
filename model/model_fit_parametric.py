from data.data_util import *
from sklearn import linear_model
from sklearn.model_selection import train_test_split, cross_validate
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix, classification_report
import joblib

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

# find the columns that are dummy variables - we do not want to scale these
num_cols = [c for c in model_features if not np.isin(model_features[c].dropna().unique(), [-1, 0, 1]).all()]

# scale the test and training set
x_train_scaled, x_test_scaled, fit_train = scale_variables(x_train, x_test, scale_cols=num_cols)

# lasso to reduce dimension
lasso_c = linear_model.SGDClassifier(
    loss='log',
    penalty='elasticnet',
    l1_ratio=.2,
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
lasso_cv_metric_dict = cross_validate(
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

# subset original frame to see what the columns are
non_zero_idx = list(np.nonzero(lasso_weights)[1])
best_features_list = x_train.iloc[:, non_zero_idx].columns
df_best_features = model_features[best_features_list]
df_best_features.mean(axis=0)

# subset features and re-fit
x_train_best = x_train_scaled[best_features_list]
x_test_best = x_test_scaled[best_features_list]

# fit a vanilla least squares model on the best coefficients
least_squares = linear_model.LogisticRegression(
    penalty='none',
    fit_intercept=True,
    class_weight={0: 1, 1: 100},
    max_iter=2000
)

# fit and extract
least_squares_fit = least_squares.fit(x_train_best, y_train)
least_squares_weights = least_squares_fit.coef_
print(f"MODEL SCORE: {least_squares_fit.score(x_train_best, y_train)}")

# cross validate the regression
ls_cv_metric_dict = cross_validate(
    least_squares_fit,
    X=x_train_best,
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

# evaluate model
y_pred = least_squares_fit.predict(x_train_best)
roc_score = roc_auc_score(y_train, y_pred)
recall_score_ = recall_score(y_train, y_pred.round())
precision_score_ = precision_score(y_train, y_pred.round())

print(f"PREDICTED ACCEPTED: {np.mean(y_pred)}")
print(f"AUC ROC SCORE: {roc_score}")
print(f"RECALL SCORE: {recall_score_}")
print(f"PRECISION SCORE: {precision_score_}")
print(confusion_matrix(y_train, y_pred.round()))
print(classification_report(y_train, y_pred.round()))

# build a coefficient dictionary
ls_dict = {}
for i, v in enumerate(list(least_squares_weights[0])):
    print(i, v)
    if abs(v) >= 0:
        print(v)
        name = x_train_best.iloc[:, i].name
        ls_dict[name] = v
print(ls_dict)

# convert dict to frame for plotting
ls_df = pd.DataFrame.from_dict(ls_dict, orient='index').reset_index()
ls_df.columns = ['variable', 'beta']
ls_df['variable'] = ls_df['variable'].str.replace('_', ' ').str.replace('flag', '')

plt.rcParams['figure.figsize'] = (12, 8)
sns.barplot(
    y='variable',
    x='beta',
    data=ls_df[abs(ls_df['beta']) >= 0.0001].sort_values('beta', ascending=False),
    palette='Blues_r'
).set_title('Linear Model Weights: Feature impact on Log Odds of Sales Acceptance')
plt.axvline(np.median(ls_df['beta']), linestyle='--', color='orange')
plt.ylabel('')
plt.xlim(-2.5, 2.5)
plt.show()

# save models
model_names = [
    './data/pickles/logistic_model.sav',
    './data/pickles/sgdc_model.sav',
    './data/pickles/parametric_model_scaler.sav'
]
model_objects = [least_squares, lasso_c, fit_train]
for i, v in enumerate(model_names):
    joblib.dump(model_objects[i], v)
