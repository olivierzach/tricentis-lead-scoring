from data.data_util import *
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix, classification_report
import joblib
from sklearn.ensemble import GradientBoostingClassifier
import skopt

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

# initialize the gbm classifier
model_gbm = GradientBoostingClassifier(
    learning_rate=0.01,
    loss='deviance',
    max_depth=200,
    max_features='sqrt',
    min_impurity_decrease=0.0,
    min_impurity_split=None,
    min_samples_leaf=5,
    min_samples_split=5,
    n_estimators=400,
    n_iter_no_change=10,
    presort='auto',
    random_state=758491,
    subsample=0.3,
    tol=0.001,
    validation_fraction=0.3,
    verbose=10,
    warm_start=False
)

# fit the classifier
model_gbm_fit = model_gbm.fit(x_train_scaled, y_train)

# cross validate the gbm model
gbm_cv_metrics_dict = cross_validate(
    model_gbm_fit,
    X=x_train_scaled,
    y=y_train,
    cv=5,
    scoring='roc_auc'
)

# get prob predictions
prob_predictions = model_gbm_fit.predict_proba(x_train_scaled)[:, 1]

# examine distribution and key moments
sns.kdeplot(prob_predictions)
plt.show()

# key moments of the distribution
percentiles = [1, 25, 50, 75, 90, 99]
prob_predictions_percentiles = np.percentile(prob_predictions, percentiles)

for i in prob_predictions_percentiles:
    count_ = np.sum(np.where(prob_predictions > i, 1, 0))
    share_ = np.mean(np.where(prob_predictions > i, 1, 0))
    print(f"PERCENTILE {i}: {count_} COUNT, {share_} SHARE")

# find the optimal threshold values
possible_thresholds = np.linspace(start=0, stop=.2, num=10000)
score_dict = {}
for i in possible_thresholds:
    y_pred = np.where(prob_predictions > i, 1, 0)
    roc_score = roc_auc_score(y_train, y_pred)
    recall_score_ = recall_score(y_train, y_pred.round())
    precision_score_ = precision_score(y_train, y_pred.round())

    score_dict[i] = [recall_score_, precision_score_]

# extract into a dict
score_df = pd.DataFrame.from_dict(score_dict, orient='index').reset_index()
score_df.columns = ['threshold', 'recall', 'precision']
score_df['diff'] = abs(score_df['recall'] - score_df['precision'])
print(score_df.sort_values('diff').iloc[0, :])

# plot the trade-offs
score_df_plot = score_df.melt(id_vars='threshold', value_vars=['recall', 'precision'])
sns.lineplot(
    y='value',
    x='threshold',
    data=score_df_plot,
    hue='variable'
).set_title('GBM Model: Classification Threshold Analysis')
plt.axvline(.063166, linestyle='--', color='grey')
plt.show()

# evaluate model
classification_threshold = .033166
y_pred = np.where(prob_predictions > classification_threshold, 1, 0)
roc_score = roc_auc_score(y_train, y_pred)
recall_score_ = recall_score(y_train, y_pred.round())
precision_score_ = precision_score(y_train, y_pred.round())

print(f"PREDICTED ACCEPTED: {np.mean(y_pred)}")
print(f"AUC ROC SCORE: {roc_score}")
print(f"RECALL SCORE: {recall_score_}")
print(f"PRECISION SCORE: {precision_score_}")
print(confusion_matrix(y_train, y_pred.round()))
print(classification_report(y_train, y_pred.round()))

# save models
model_names = [
    './data/pickles/gbm_model.sav',
    './data/pickles/gbm_scaler.sav',
    './data/pickles/gbm_classification_threshold.sav'
]
model_objects = [model_gbm, fit_train, classification_threshold]
for i, v in enumerate(model_names):
    joblib.dump(model_objects[i], v)
