import pandas as pd
import numpy as np
import re
from keras import backend as K
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
import matplotlib.pyplot as plt
import seaborn as sns


def dummy_wrapper(df, cols_to_dummy=None):
    """
    Wrapper for pd.get_dummies that appends dummy variables back onto original dataset, cleans columns
    Parameters
    ----------
    df : DataFrame
    cols_to_dummy : list
    Returns
    -------
    DataFrame
    """

    df = df.copy()

    df_dummy = pd.get_dummies(df[cols_to_dummy], dummy_na=True)

    # clean the categorical column names
    df_dummy.columns = df_dummy.columns. \
        str.strip(). \
        str.lower(). \
        str.replace(' ', '_'). \
        str.replace('-', ''). \
        str.replace('/', ''). \
        str.replace('$', ''). \
        str.replace(',', ''). \
        str.replace('&', ''). \
        str.replace('.', ''). \
        str.replace('+', ''). \
        str.replace(':', ''). \
        str.replace('|', ''). \
        str.replace('[', ''). \
        str.replace(']', ''). \
        str.replace('(', '').str.replace(')', '')

    cols_to_keep = [c for c in df.columns if c not in cols_to_dummy]

    df_keep = df[cols_to_keep]

    df_clean = pd.concat([df_keep, df_dummy], axis=1)

    return df_clean


def generic_email(df, email_field):
    """
    Parses out email strings returns a categorical variable related to missing, generic, or non-generic email
    Parameters
    ----------
    df : DataFrame
    email_field : str
    Returns
    -------
    DataFrame
    """

    # list of generic emails
    generics = [
        'gmail.com',
        'yahoo.com',
        'outlook.com',
        'aol.com',
        'msn.com',
        'hotmail.com',
        'cox.net',
        'att.net',
        'sbcglobal.net'
    ]

    # generic email parsing
    df[email_field] = np.where(
        df[email_field].isnull(),
        'no_email',
        np.where(
            df[email_field].str.contains('|'.join(generics)),
            'generic_email',
            'non_generic_email'
        )
    )

    return df


def url_parsing_business(df, url_field):
    """
    Parses out url strings returns a categorical variable related to domain business grouping
    Parameters
    ----------
    df : DataFrame
    url_field : str
    Returns
    -------
    DataFrame
    """

    df[url_field] = np.where(
        df[url_field].str.contains('.com'),
        'commercial',
        np.where(
            df[url_field].str.contains('.edu'),
            'education',
            np.where(
                df[url_field].str.contains('.biz'),
                'business',
                np.where(
                    df[url_field].str.contains('.net'),
                    'network',
                    np.where(
                        df[url_field].str.contains('.DE'),
                        'germany',
                        np.where(
                            df[url_field].str.contains('.UK'),
                            'united_kingdom',
                            np.where(
                                df[url_field].str.contains('.us'),
                                'united_states',
                                np.where(
                                    df[url_field].str.contains('org'),
                                    'organization',
                                    np.where(
                                        df[url_field].isnull(),
                                        'no_domain',
                                        'domain_other'
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )

    return df


def variance_threshold(df, threshold=.001):
    """
    Checks model frame for numeric columns with zero variance and variance
    less than a provided threshold
    Parameters
    ----------
    df : DataFrame
    threshold: float
    Returns
    -------
    DataFrame
    """

    print(f"number of features before filter {df.shape[1]}")

    var_dict = np.var(df, axis=0).to_dict()

    var_dict = {k: v for (k, v) in var_dict.items() if v >= threshold}

    keep_list = list(var_dict.keys())

    df = df[keep_list]

    print(f"number of features remaining {df.shape[1]}")

    return df


def clean_categorical(df):
    """
    Simple parser to exclude random categorical column characters from the final value
    Parameters
    ----------
    df : DataFrame
    Returns
    -------
    DataFrame
    """

    df = df.str.strip(). \
        str.lower(). \
        str.replace(' ', '_'). \
        str.replace('-', ''). \
        str.replace('/', ''). \
        str.replace('$', ''). \
        str.replace(',', ''). \
        str.replace('&', ''). \
        str.replace('.', ''). \
        str.replace('|', ''). \
        str.replace('(', '').str.replace(')', '')

    return df


def clean_multi_index_headers(df):
    """
    Concatenates a multi-index columns headers into one with a clean format
    Parameters
    ----------
    df : DataFrame
    Returns
    -------
    DataFrame
    """

    df.columns = ['_'.join(col).strip() for col in df.columns.values]

    return df


def is_valid_email(email=None):
    """Validates whether the email provided is syntactically correct."""

    r = re.compile(
        """(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])"""
    )

    try:
        is_valid = False if re.match(r, email) is None else True

    except Exception as e:
        print(e)
        is_valid = False

    return is_valid


def get_path_features(path_cols, keywords, df, col_title, path_length):
    df_path = df[path_cols]
    df_path = df_path[df_path[path_cols[1]] <= path_length]

    # pivot the url path
    df_path = df_path.pivot_table(
        values=path_cols[2],
        columns=path_cols[1],
        index=path_cols[0],
        aggfunc='first',
        fill_value='None'
    )
    df_path.columns = [col_title + 'path_slot_' + str(c) for c in df_path.columns]

    # extract the slot columns
    slot_columns = df_path.columns

    # fill in the url slots with the page type
    for i in keywords:
        for j in slot_columns:

            if i == 'https://www.tricentis.com/':
                # features to show the type of url for each slot in path
                df_path[j + '_home_page'] = np.where(
                    df_path[j] == i,
                    1,
                    0
                )

            else:
                # features to show the type of url for each slot in path
                df_path[j + '_' + i] = np.where(
                    df_path[j].str.lower().str.contains(i, case=False),
                    1,
                    0
                )

    # fill in urls with an indicator of hitting specific slot in path
    for i in slot_columns:
        df_path[i] = np.where(
            df_path[i] != 'None',
            1,
            0
        )

    # length of path feature
    df_path['length_of_path_' + col_title] = df_path[slot_columns].sum(axis=1)

    # drop the slot columns
    df_path.drop(slot_columns, axis=1, inplace=True)

    return df_path


def get_stop_words():

    stop_words_list = [
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
        "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its",
        "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom",
        "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have",
        "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or",
        "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between",
        "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in",
        "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when",
        "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
        "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will",
        "just", "don", "should", "now"
    ]

    return stop_words_list


def clean_string(s):

    s = s.lower(). \
        replace('/', ''). \
        replace('(', ''). \
        replace(')', ''). \
        replace('-', ''). \
        replace(' ', '_'). \
        replace('3', 'three')

    return s


def focal_loss(y_true, y_pred):
    gamma = 2.0
    alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha*K.pow(1.0-pt_1, gamma)*K.log(pt_1))-K.sum((1-alpha) * K.pow(pt_0, gamma)*K.log(1.0-pt_0))


def scale_variables(train, test, scale_cols):
    """
    Applies StandardScalar() learned on the training dataset to the training and test data frames
    Applying the learned scalar on train to the test frames helps avoid information leakage
    Parameters
    ----------
    train : training data frame of features
    test : testing data frame of features
    scale_cols : list of columns we want to scale helpful to avoid scaling binary flags
    Returns
    -------
    DataFrame : scaled training and test set data frames
    fit object: StandardScaler() fit object to use for future predictions
    """

    selector = StandardScaler()

    # subset train and test into scale columns and not scale columns
    df_scale_train = train[scale_cols]
    df_scale_test = test[scale_cols]
    df_not_scale_train = train.drop(scale_cols, axis=1)
    df_not_scale_test = test.drop(scale_cols, axis=1)

    # scale the train set
    df_scaled_train = pd.DataFrame(
        selector.fit_transform(df_scale_train),
        columns=df_scale_train.columns,
        index=df_scale_train.index
    )

    # scale the test set using scaler fit on test
    df_scaled_test = pd.DataFrame(
        selector.transform(df_scale_test),
        columns=df_scale_test.columns,
        index=df_scale_test.index
    )

    # combined scaled and non-scaled back together
    df_train_complete = pd.concat([df_not_scale_train, df_scaled_train], axis=1)
    df_test_complete = pd.concat([df_not_scale_test, df_scaled_test], axis=1)

    return (
        df_train_complete,
        df_test_complete,
        selector.fit(train)
    )

def extra_trees_vimp(
        df,
        y,
        threshold=.01,
        plot=True,
        estimators=100,
        depth=3,
        split_sample=.05,
        leaf_sample=.05,
        transform=True
):

    print('Building Trees...')

    x_vars = df
    y_vars = y

    # flow control for regression or classification
    regression_type = y_vars.drop_duplicates()

    if len(regression_type) == 2:

        print('Building Classification Trees...')

        model = ExtraTreesClassifier(
            n_estimators=estimators,
            max_depth=depth,
            random_state=444,
            min_samples_split=split_sample,
            min_samples_leaf=leaf_sample,
            class_weight='balanced_subsample',
            max_features='log2',
            bootstrap=True,
            oob_score=True
            )

        model.fit(x_vars, np.asarray(y_vars).ravel())

        importance = model.feature_importances_

        df = pd.DataFrame(importance)
        df = df.T
        df.columns = x_vars.columns
        df = df.T.reset_index()
        df.columns = ['variable', 'tree_vimp']
        df = df.sort_values('tree_vimp', ascending=False)

    else:

        if transform:
            y_vars = np.sqrt(y_vars)

        print('Building Regression Trees...')

        model = ExtraTreesRegressor(
            n_estimators=estimators,
            max_depth=depth,
            random_state=444,
            min_samples_split=split_sample,
            min_samples_leaf=leaf_sample,
            max_features='log2',
            bootstrap=True,
            oob_score=True
            )
        model.fit(x_vars, np.asarray(y_vars).ravel())

        importance = model.feature_importances_

        df = pd.DataFrame(importance)
        df = df.T
        df.columns = x_vars.columns
        df = df.T.reset_index()
        df.columns = ['variable', 'tree_vimp']
        df = df.sort_values('tree_vimp', ascending=False)

    if plot:
        plt.figure()
        sns.barplot(
            x='tree_vimp',
            y='variable',
            data=df[df.tree_vimp >= threshold],
            palette='Blues_r',
        ).set_title(y.name)

    # extract the best tree importance results
    df = df[df.tree_vimp >= threshold]
    important_cols = list(df.variable)

    print('Tree Models Complete')

    return df, important_cols, model.oob_score_

