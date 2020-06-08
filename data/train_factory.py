from os import listdir
from os.path import isfile, join
import pandas as pd

# extract locations of pickled data files
pickle_path = './data/pickles/'
pickle_loc_list = [f for f in listdir(pickle_path) if isfile(join(pickle_path, f))]

# read in all the pickled data frames
data_assets = {}
for i in pickle_loc_list:
    data_assets[i[:-4]] = pd.read_pickle(pickle_path + i)

# extract frames from dict
df_targets = data_assets['target_data']
df_leads = data_assets['leads_data'].reset_index()
df_tp = data_assets['touchpoints_data']
df_hgi = data_assets['hgi_data']
df_disc_org = data_assets['discover_org_data'].drop_duplicates()

# TODO: need to understand keys for HGI and DO data
# TODO: double check these are not input after becoming a prospect "passed to sales"

# merge all frames together with leads as a base
df = (
    df_leads.merge(
        df_targets,
        how='left',
        left_on='email',
        right_on='email'
    ).merge(
        df_tp,
        how='left',
        left_on='email',
        right_on='EMAIL_ADDRESS'
    ).merge(
        df_disc_org,
        how='left',
        left_on='email_domain',
        right_on='company_email_domain'
    )
).fillna(0).drop_duplicates()