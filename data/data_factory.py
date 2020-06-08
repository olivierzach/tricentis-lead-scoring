from data.profile_leads_data import *
from data.profile_touch_point_data import *
from data.profile_hgi_data import *
from data.profile_discover_org_data import *
from data.profile_target_data import *
import pickle


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
    df, global_map_dict = profile_discover_org_data(df)
    df.to_pickle(pickle_name)

    # pickle the look up imputation dict for high cardinality data
    with open(pickle_name[:15] + 'discover_map_dict.pkl', 'wb') as handle:
        pickle.dump(global_map_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return df, global_map_dict


def source_model_targets(
    file_name="./data/source/Leads_data.csv",
    pickle_name="./data/pickles/target_data.pkl"
):
    df = pd.read_csv(file_name, engine='python')
    df = profile_target_data(df)
    df.to_pickle(pickle_name)

    return df

source_leads_data()
