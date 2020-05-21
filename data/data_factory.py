import pandas as pd
import pickle

# TODO: read in an clean data for each dataset


def source_account_data(
    file_name="./data/source/Account_data.csv",
    pickle_name="./data/pickles/account_data.pkl"
    ):

    df = pd.read_csv(file_name)
    df.to_pickle(pickle_name)


def source_opportunities_data(
    file_name="./data/source/Opportunities_data.csv",
    pickle_name="./data/pickles/opportunities_data.pkl"
    ):

    df = pd.read_csv(file_name)
    df.to_pickle(pickle_name)


def source_leads_data(
    file_name="./data/source/Leads_data.csv",
    pickle_name="./data/pickles/leads_data.pkl"
    ):

    df = pd.read_csv(file_name)
    df.to_pickle(pickle_name)


def source_touchpoints_data(
    file_name="./data/source/Touchpoints_data.csv",
    pickle_name="./data/pickles/touchpoints_data.pkl"
    ):

    df = pd.read_csv(file_name)
    df.to_pickle(pickle_name)


def source_hgi_data(
    file_name="./data/source/HGI_data.csv",
    pickle_name="./data/pickles/hgi_data.pkl"
    ):

    df = pd.read_csv(file_name)
    df.to_pickle(pickle_name)


def source_discover_org_data(
    file_name="./data/source/DiscoverOrg_data.csv",
    pickle_name="./data/pickles/discover_org_data.pkl"
    ):
    
    df = pd.read_csv(file_name)
    df.to_pickle(pickle_name)
