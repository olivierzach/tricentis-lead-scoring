from data.data_util import *
pd.options.mode.chained_assignment = None


def profile_hgi_data(df):

    # available columns
    feature_cols = [
        'account_employee_range',
        'account_revenue_range',
        'account_industry',
        'account_sub_industry',
        'product_vendor',
        'product_name'
    ]

    # clean the categorical columns
    for i in feature_cols:
        df[i] = clean_categorical(df[i])

    # dummy out variables
    df = dummy_wrapper(df, cols_to_dummy=feature_cols)

    # roll up to the account level
    df = df.groupby('account_id').max()

    # filter out low variance features
    for i in df.columns:
        if np.var(df[i]) < .05:
            df.drop(i, axis=1, inplace=True)

    # total amount of features available
    df['hgi_coverage'] = df.mean(axis=1)

    # expose the account id back to join to other tables
    df.reset_index(inplace=True)

    return df
