import pandas as pd
import numpy as np
import re


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


def get_path_features(path_cols, keywords, df, col_title):
    df_path = df[path_cols]
    df_path = df_path[df_path[path_cols[1]] <= 10]

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
