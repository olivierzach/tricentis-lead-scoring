from data.data_util import *

df = pd.read_csv("./data/source/Touchpoints_data.csv", engine='python')

# TODO: content url base features
# TODO: group by and sum the base features by email to join to leads table
# TODO: join in the web page, channel, and content path features
# TODO: use the leads table to find the censoring date for all path features
# TODO: put this in a function
# TODO: profile the join to leads data, make sure to account for leads with missing connections

# drop the ignore columns
drop_cols = [
    'ID',
    'ANN_TOUCHPOINT_ID__MD5_',
    'EMAIL_ADDRESS__MD5_',
    'PAGEVISIT_DATETIME',
    'SOURCE',
    'SFDC_CAMPAIGN_ID__15_',
    'TACTIC',
    'CAMPAIGN',
    'PPC_AD',
    'PPC_AD_GROUP',
    'PERIOD',
    'ABBR',
    'SUBSTANTIVE',
    'co_code'
]
df.drop(drop_cols, axis=1, inplace=True)

# convert touch point date
df['TOUCHPOINT_DATE'] = pd.to_datetime(df['TOUCHPOINT_DATE'])

# year and month features
df['touch_point_month'] = df['TOUCHPOINT_DATE'].dt.month
df['touch_point_year'] = df['TOUCHPOINT_DATE'].dt.year

feature_keywords = [
    'webinar',
    'referral',
    'acceler',
    'none',
    'nurture',
    'direct',
    'inbound',
    'partner',
    'email',
    'organic'
]

for i in feature_keywords:
    df['channel_' + i] = np.where(
        df.CHANNEL.str.contains(i, case=False),
        1,
        0
    )

# digital marketing flag
df['channel_digital_marketing'] = np.where(
    df.CHANNEL.str.contains('ppc|search|seo|paid', case=False, regex=True),
    1,
    0
)

# channel features
keep_channels = [
    'website',
    'content syndication',
    'direct mail',
    'tradeshow',
    'webinar syndication',
    'house list',
    'search',
    'event field',
    'ppc',
    'webinar organic',
    'direct website',
    'inbound email',
    'customer event',
    'inbound phone',
    'organic website',
    'social paid',
    'seo',
    'event virtual',
    'qualityjam'
]

# append detailed channel descriptions
for i in keep_channels:
    name = i.replace(" ", "")
    df['channel_' + name] = np.where(
        df.CHANNEL.str.lower() == i,
        1,
        0
    )

# device type features
for i in list(set(df.DEVICE)):
    name = str(i).lower().strip()

    df['device_' + name] = np.where(
        df['DEVICE'] == i,
        1,
        0
    )

# previous page features
df['page_flag_none'] = np.where(
    df.url == 'None',
    1,
    0
)

# flags we want to develop for the web pages
page_feature_keywords = [
    'google',
    'facebook',
    'bing',
    'linkedin',
    'tricentis',
    'connect',
    'resource',
    'product',
    'demo',
    'training',
    'blog',
    'accelerate',
    'thank',
    'article'
]

# build the flags
for i in page_feature_keywords:
    df['page_flag_' + i] = np.where(
        df.PREVIOUS_PAGE.str.contains(i, case=False),
        1,
        0
    )

# explicit missing web page flag
df['page_flag_na'] = np.where(
    df.PREVIOUS_PAGE.isna(),
    1,
    0
)

# actions
action_keywords = ['form', 'register', 'attended']
for i in action_keywords:
    df['action_flag_' + i] = np.where(
        df.ACTION.str.lower().str.contains("form", case=False),
        1,
        0
    )

# explicit missing action flag
df['action_flag_na'] = np.where(
    df.ACTION.isna(),
    1,
    0
)

# define the url path parameters
path_cols = ['EMAIL_ADDRESS', 'ACTIVITY_ORDER', 'PREVIOUS_PAGE']
col_title = 'url'
keywords = [
    'google',
    'facebook',
    'bing',
    'linkedin',
    'sap',
    'events',
    'academy',
    'connect',
    'resource',
    'products',
    'demo',
    'blog',
    'accelerate',
    'article',
    'csod',
    'agenda',
    'contact',
    'community',
    'enterprise',
    'https://www.tricentis.com/'
]


# get the pivoted url path to join to original frame
df_url_path = get_path_features(
    path_cols=path_cols,
    keywords=keywords,
    df=df,
    col_title=col_title
)

# define the content path parameters
path_cols = ['EMAIL_ADDRESS', 'CONTENT_ORDER', 'final_title']
col_title = 'content'
keywords = [
    'accelerate',
    'forrester',
    'paper',
    'sap',
    'agile',
    'devops',
    'test',
    'gartner',
    'web',
    'automation',
    'report',
    'qa',
    'roadshow',
    'ai'
    'quality',
    'rpa',
    'demo',
    'transformation',
    'management',
    'book'
]

# get the pivoted content path to join to original frame
df_content_path = get_path_features(
    path_cols=path_cols,
    keywords=keywords,
    df=df,
    col_title=col_title
)
