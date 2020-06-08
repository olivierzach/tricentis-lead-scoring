from data.data_util import *
from keras import models
from keras import layers
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

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

# sample data for an unbiased look at the model
df_model = df_model_base.sample(frac=.85)

# split out targets and features - heavily imbalanced labels
model_target_name = 'accepted_by_sales'
model_features = df_model.drop(model_target_name, axis=1)
model_target = df_model[model_target_name].values

# kill columns near zero variance
model_features = variance_threshold(model_features, .05)

# cross validation splits
x_train, x_test, y_train, y_test = train_test_split(
    model_features,
    model_target,
    test_size=.2,
    shuffle=True,
    stratify=model_target,
    random_state=7651244
)

# feature normalization - compute normalization on the training data only
x_train_mean = x_train.mean(axis=0)
x_train_std = x_train.std(axis=0)

# normalize the training set
x_train -= x_train_mean
x_train /= x_train_std

# apply training normalization on test to avoid leakage
x_test -= x_train_mean
x_test /= x_train_std

# define our model
model = models.Sequential()
model.add(layers.Dense(20, activation='relu', input_shape=(x_train.shape[1], ), kernel_initializer='he_uniform'))
model.add(layers.Dense(15, activation='relu'))
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(5, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# choosing a loss function and an optimizer
# binary classification: binary cross entropy
model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# model training
weights = {0: 1, 1: 99}
fit = model.fit(
    x_train,
    y_train,
    epochs=20,
    batch_size=100,
    class_weight=weights
)

# evaluate model
y_pred = model.predict(x_train)
score = roc_auc_score(y_train, y_pred)
print(f"PREDICTED ACCEPTED: {np.mean(y_pred)}")
print(f"AUC ROC SCORE: {score}")

# save model
save_path = './data/pickles/model_dnn_sales_accepted'
model.save(save_path)
