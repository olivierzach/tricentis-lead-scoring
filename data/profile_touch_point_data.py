from data.data_util import *

df = pd.read_csv("./data/source/Touchpoints_data.csv", engine='python')

# TODO: content url base features
# TODO: how best to encode content without making model too frail
# TODO: group by and sum the base features by email to join to leads table
# TODO: join in the web page, channel, and content path features
# TODO: use the leads table to find the censoring date for all path features
# TODO: put this in a function
# TODO: profile the join to leads data, make sure to account for leads with missing connections
# TODO: can we get timestamp for page and content activity?

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

# strip all columns
for i in df.columns:
    try:
        df[i] = df[i].apply(lambda x: x.strip())
    except Exception as e:
        print(e)

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

# if previous page null, fill in with current url
df['PREVIOUS_PAGE'] = np.where(
    (df['PREVIOUS_PAGE'].isnull()) | (df['PREVIOUS_PAGE'] is not None) | (df['PREVIOUS_PAGE'] == 'None'),
    df['url'],
    df['PREVIOUS_PAGE']
)

# define the url path parameters = themes of the touch point without getting too granular
path_cols = ['EMAIL_ADDRESS', 'ACTIVITY_ORDER', 'PREVIOUS_PAGE']
col_title = 'url'
keywords = [
    'academy',
    'accelerate',
    'agenda',
    'article',
    'become',
    'bing',
    'blog',
    'book',
    'community',
    'compare',
    'connect',
    'contact',
    'csod',
    'customer-experience',
    'demo',
    'enterprise',
    'events',
    'facebook',
    'google',
    'humana',
    'https://www.tricentis.com/',
    'integrations',
    'linkedin',
    'locations',
    'partner',
    'portal',
    'products',
    'resource',
    'sap',
    'solutions',
    'routenamelearning',
    'tosca',
    'qasymphony',
    'qtestnet'
]


# get the pivoted url path to join to original frame
df_url_path = get_path_features(
    path_cols=path_cols,
    keywords=keywords,
    df=df,
    col_title=col_title,
    path_length=9
)


# define the content path parameters = themes of the content without getting too granular
path_cols = ['EMAIL_ADDRESS', 'CONTENT_ORDER', 'final_title']
col_title = 'content'
keywords = [
    'testing',
    'accelerate',
    'automation',
    'sap',
    'software',
    'devops',
    'agile',
    'quality',
    'continous',
    'tricentis',
    'transformation',
    'qa',
    'digital',
    'management',
    'future',
    'research',
    'success',
    'business',
    'roadshow',
    'ai',
    'tosca',
    'warehouse',
    'forrester',
    'challenges',
    'gartner',
    'webinar',
    'migration',
    'enterprise',
    'code',
    'virtualization',
    'suite',
    'risk',
    'book',
    'paper'
]

# get the pivoted content path to join to original frame
df_content_path = get_path_features(
    path_cols=path_cols,
    keywords=keywords,
    df=df,
    col_title=col_title,
    path_length=9
)

# if previous channel null, fill in with current channel
df['PREVIOUS_CHANNEL'] = np.where(
    (df['PREVIOUS_CHANNEL'].isnull()) | (df['PREVIOUS_CHANNEL'] is not None) | (df['PREVIOUS_CHANNEL'] == 'None'),
    df['CHANNEL'],
    df['PREVIOUS_CHANNEL']
)

# define the content path parameters = themes of the content without getting too granular
path_cols = ['EMAIL_ADDRESS', 'CONTENT_ORDER', 'PREVIOUS_CHANNEL']
col_title = 'content'
keywords = [
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

# get the pivoted content path to join to original frame
df_channel_path = get_path_features(
    path_cols=path_cols,
    keywords=keywords,
    df=df,
    col_title=col_title,
    path_length=9
)

# time based features
time_cols = ['EMAIL_ADDRESS', 'TOUCHPOINT_DATE']
df_time = df[time_cols]

# different touch point date by email = time between events
df_time['time_diff'] = df_time.sort_values(time_cols)\
    .groupby('EMAIL_ADDRESS')['TOUCHPOINT_DATE']\
    .diff()

# result is a time delta object that we can convert to days
df_time['time_diff'] = df_time['time_diff'].apply(lambda x: x.total_seconds() / (60*60*24))

# roll up to one row per email, grabbing time based features on the way
df_time = df_time.groupby('EMAIL_ADDRESS').agg(['mean', 'std', 'sum', 'count'])
df_time = clean_multi_index_headers(df_time)
df_time.reset_index(inplace=True)


for i in df.columns:
    print(df[i].value_counts())

# content attribute features - dummy out the content flag
content_type_cols = ['EMAIL_ADDRESS', 'content_type_tag']
df_content_type = dummy_wrapper(df[content_type_cols], cols_to_dummy='content_type_tag')

# very sparse categories - need to bin into larger buckets
# accelerate group
df_content_type['content_type_accelerate'] = (
    df_content_type['accelerate_on_demand'] + df_content_type['accelerator']
)

# video group
df_content_type['content_type_video'] = (
    df_content_type['video'] +
    df_content_type['demo_video']
)

# live content
df_content_type['content_type_live'] = (
    df_content_type['event'] +
    df_content_type['webinar']
)

# static content
df_content_type['content_type_static'] = (
    df_content_type['book'] +
    df_content_type['ebook'] +
    df_content_type['paper'] +
    df_content_type['white_paper'] +
    df_content_type['guides__insights']
)

# awareness content it typically read or learn more materials about the problem business solves
df_content_type['content_type_awareness'] = (
    df_content_type['book'] +
    df_content_type['event'] +
    df_content_type['paper'] +
    df_content_type['guides__insights'] +
    df_content_type['ebook'] +
    df_content_type['white_paper']
)

# acquisition content is geared towards features and selling the product
df_content_type['content_type_acquisition'] = (
    df_content_type['accelerate_on_demand'] +
    df_content_type['accelerator'] +
    df_content_type['demo_video'] +
    df_content_type['video'] +
    df_content_type['webinar'] +
    df_content_type['analyst_research']
)

# group by mean to not leak total counts into path variables
df_content_type = df_content_type.groupby('EMAIL_ADDRESS').mean()






