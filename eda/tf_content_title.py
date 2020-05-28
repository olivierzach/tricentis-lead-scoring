from data.data_util import *
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

# read in our data
df = pd.read_csv("./data/source/Touchpoints_data.csv", engine='python')

# extract the touch point urls
keep_cols = ['final_title']
df = df[keep_cols].dropna().drop_duplicates()

# convert frame to dict
doc_dict = {}
for i in range(df.shape[0]):

    # clean up the input web page urls
    doc = df.iloc[i, :].values[0]\
        .replace('/', ' ')\
        .replace('.', ' ')\
        .replace('=', '')\
        .replace('?', '')\
        .replace(':', '')\
        .replace('-', ' ')\
        .lower()

    doc_dict['doc_' + str(i)] = doc

# corpus of web page strings
corpus = [v for v in doc_dict.values()]

# initialize the module
vector = CountVectorizer()

# transform each of the documents into a tf vector
vector_fit = vector.fit_transform(corpus)

# grab the feature names and the list of tf outputs
feature_names = vector.get_feature_names()
dense_list = vector_fit.todense().tolist()

# stack in a frame for inspection
df_tf = pd.DataFrame(dense_list, columns=feature_names)

# see the tf for all words found
df_calc = df_tf.mean(axis=0).reset_index()
df_calc.columns = ['string', 'tf']

# filter to only readable strings
df_calc = df_calc[df_calc.string.apply(lambda x: x.isalpha())]\
    .sort_values('tf', ascending=False)

# manually remove some common stop words
stop_words = get_stop_words()
df_calc = df_calc[~df_calc.string.isin(stop_words)]

# plot the results
plt.figure(figsize=(12, 8))
sns.barplot(
    x='tf',
    y='string',
    data=df_calc.iloc[:75, :]
).set_title('Content Title Term Frequency')
plt.show()
