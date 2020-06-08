from data.data_util import *
pd.options.mode.chained_assignment = None


def profile_target_data(df):

    # select the target columns from the dataset
    keep_cols = [
        'email',
        'email_domain',
        'passed_to_sales',
        'accepted_by_sales',
        'opportunity_won'
    ]
    df = df[keep_cols].drop_duplicates()

    # format bool targets
    bool_targets = [
        'passed_to_sales',
        'accepted_by_sales',
        'opportunity_won'
    ]

    for i in bool_targets:
        df[i] = df[i].fillna(0).astype(int)

    return df
