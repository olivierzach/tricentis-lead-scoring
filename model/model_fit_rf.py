from data.data_util import *
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix, classification_report
import joblib
from sklearn.model_selection import GridSearchCV

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

# initialize the model
forest_c = ExtraTreesClassifier(
    n_estimators=200,
    criterion='entropy',
    max_depth=200,
    min_samples_leaf=5,
    min_samples_split=5,
    max_features='log2',
    bootstrap=True,
    random_state=4312774,
    oob_score=True,
    class_weight={0: 1, 1: 100},
    max_samples=5000,
    ccp_alpha=.1
)

# fit and extract
forest_fit = forest_c.fit(x_train_scaled, y_train)
print(f"MODEL SCORE: {forest_fit.score(x_train_scaled, y_train)}")
print(f"MODEL OOB SCORE: {forest_fit.oob_score_}")

# cross validate the forest
forest_cv_metrics_dict = cross_validate(
    forest_fit,
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

# evaluate model
y_pred_prob = forest_fit.predict_proba(x_train_scaled)[:, 1]
roc_score = roc_auc_score(y_train, y_pred_prob)
recall_score_ = recall_score(y_train, y_pred_prob.round())
precision_score_ = precision_score(y_train, y_pred_prob.round())

print(f"PREDICTED ACCEPTED: {np.mean(y_pred_prob.astype(int))}")
print(f"AUC ROC SCORE: {roc_score}")
print(f"RECALL SCORE: {recall_score_}")
print(f"PRECISION SCORE: {precision_score_}")
print(confusion_matrix(y_train, y_pred_prob.round()))
print(classification_report(y_train, y_pred_prob.round()))

# examine distribution and key moments
sns.kdeplot(y_pred_prob)
plt.show()

# key moments of the distribution
percentiles = [1, 25, 50, 75, 90, 99]
prob_predictions_percentiles = np.percentile(y_pred_prob, percentiles)

# key percentiles of predictions
for i in prob_predictions_percentiles:
    count_ = np.sum(np.where(y_pred_prob > i, 1, 0))
    share_ = np.mean(np.where(y_pred_prob > i, 1, 0))
    print(f"PERCENTILE {i}: {count_} COUNT, {share_} SHARE")

# find the optimal threshold values
possible_thresholds = np.linspace(start=.48, stop=.50, num=100)
score_dict = {}
for i in possible_thresholds:
    y_pred = np.where(y_pred_prob > i, 1, 0)
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
).set_title('RF Model: Classification Threshold Analysis')
plt.show()

# analyze the feature importance
importance = forest_c.feature_importances_
df_vi = pd.DataFrame(importance)
df_vi = df_vi.T
df_vi.columns = x_train_scaled.columns
df_vi = df_vi.T.reset_index()
df_vi.columns = ['variable', 'tree_vimp']
df_vi = df_vi.sort_values('tree_vimp', ascending=False)

plt.rcParams['figure.figsize'] = (12, 8)
sns.barplot(
    x='tree_vimp',
    y='variable',
    data=df_vi[df_vi.tree_vimp >= .00001],
    palette='Blues_r',
).set_title('Tree Based Model: Feature Importance for Sales Acceptance')
plt.show()

# set up grid search
weights = np.linspace(90, 120, 10).astype(int)
weights = [5, 10, 90, 97, 100, 104, 110]
opt = GridSearchCV(
    forest_c,
    {
        "class_weight": [{0: 1, 1: x} for x in weights],
        "ccp_alpha": [.1, .01, 1],
        "criterion": ['gini', 'entropy']
    },
    cv=3,
    scoring='f1_macro',
    verbose=1,
    n_jobs=-1
)

# fit parameters
rf_opt_fit = opt.fit(x_train_scaled, y_train)
print(f"BEST PARAMETERS: {rf_opt_fit.best_params_}")

# save models
model_names = ['./data/pickles/rf_model.sav', './data/pickles/rf_scaler.sav']
model_objects = [forest_c, fit_train]
for i, v in enumerate(model_names):
    joblib.dump(model_objects[i], v)
