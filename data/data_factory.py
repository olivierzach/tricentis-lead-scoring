import pandas as pd
from data.profile_leads_data import *
from data.profile_touch_point_data import *
from data.profile_hgi_data import *

# TODO: complete the profile of the touch points data
# TODO: source these in a build script and join components together


def source_leads_data(
    file_name="./data/source/Leads_data.csv",
    pickle_name="./data/pickles/leads_data.pkl"
):
    df = pd.read_csv(file_name, engine='python')
    df, df_c = profile_leads_data(df)
    df.to_pickle(pickle_name)
    df_c.to_pickle(pickle_name[:15] + 'censor_key.pkl')

    return df, df_c


def source_touch_points_data(
    file_name="./data/source/Touchpoints_data.csv",
    pickle_name="./data/pickles/touchpoints_data.pkl",
    censor_key=None
):

    df = pd.read_csv(file_name, engine='python')
    df = profile_touch_point_data(df, censor_key)
    df.to_pickle(pickle_name)

    return df


def source_hgi_data(
    file_name="./data/source/HGI_data.csv",
    pickle_name="./data/pickles/hgi_data.pkl"
):

    df = pd.read_csv(file_name, engine='python')
    df = profile_hgi_data(df)
    df.to_pickle(pickle_name)

    return df


def source_discover_org_data(
    file_name="./data/source/DiscoverOrg_data.csv",
    pickle_name="./data/pickles/discover_org_data.pkl"
):

    df = pd.read_csv(file_name, engine='python')
    df.to_pickle(pickle_name)
