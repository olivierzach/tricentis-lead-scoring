from data.data_util import *
from functools import reduce
pd.options.mode.chained_assignment = None


def profile_touch_point_data(df, df_censor_key):

    # strip all columns
    for i in df.columns:
        try:
            df[i] = df[i].apply(lambda x: x.strip())
        except Exception as e:
            print(e)

    # convert touch point date
    df['TOUCHPOINT_DATE'] = pd.to_datetime(df['TOUCHPOINT_DATE'])

    # join the the censor key from the leads table
    df = df.merge(
        df_censor_key,
        how='left',
        left_on='EMAIL_ADDRESS',
        right_on='email'
    )

    # flag if touch point greater than censor date
    df['censor_flag'] = np.where(
        df['TOUCHPOINT_DATE'] > df['opportunity_created_date'],
        1,
        0
    )

    # filter out censored touch points
    df = df[df['censor_flag'] < 1]

    # time based features
    keep_cols = ['EMAIL_ADDRESS', 'TOUCHPOINT_DATE']
    df_time = df[keep_cols]

    # roll up to max and min date by email
    df_time = df_time.groupby('EMAIL_ADDRESS').agg(['max', 'min'])
    df_time = clean_multi_index_headers(df_time)

    # time between touch points
    df_time['touch_point_diff_hours'] = (df_time['TOUCHPOINT_DATE_max'] - df_time['TOUCHPOINT_DATE_min']
                                         ).dt.seconds / 3600

    # remove the calculation columns
    drop_cols = ['TOUCHPOINT_DATE_max', 'TOUCHPOINT_DATE_min']
    df_time.drop(drop_cols, axis=1, inplace=True)

    # channel keywords
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

    # combine all features together
    dfs = [df, df_url_path, df_content_path, df_channel_path, df_content_type, df_time]
    df_merged = reduce(
        lambda left, right: pd.merge(left, right, on='EMAIL_ADDRESS', how='inner'),
        dfs
    )

    # remove all columns that we do not need
    # drop the ignore columns
    drop_cols = [
        'email',
        'ai',
        'NEXT_CHANNEL',
        'PPC_AD_GROUP',
        'n-co-4',
        'spreadsheets',
        'se-co-2',
        'devops_testing',
        'CHANNEL',
        'FORM_COUNT',
        'DOMAIN',
        'content_type_tag',
        'tricentis_academy_and_community',
        'business_transformation_(cio)',
        'PAGEVISIT_DATETIME',
        'url',
        'hp/micro_focus_migration',
        'professional_trends',
        'FORM',
        'co_code',
        'agile_test_management_',
        'agile_testing',
        'jira',
        'strategic_it',
        'rpa',
        'PREVIOUS_PAGE',
        'api_testing',
        'load_testing',
        'analytics',
        'CONTENT_ORDER',
        'ACTIVITY_ORDER',
        'pdg/nonpdg',
        'SOURCE',
        'se-co-1',
        'CAMPAIGN',
        'e-co-1',
        'NEXT_CONTENT',
        'PREVIOUS_CHANNEL',
        'tricentis_qtest',
        'testing_leadership',
        'TACTIC',
        'risk_based_testing',
        'testing_scalability_&_efficiency',
        'PPC_AD',
        'regulatory_compliance',
        'integrations',
        'DEVICE',
        'SFDC_CAMPAIGN_ID__15_',
        'test_automation',
        'tricentis_livecompare',
        'exploratory_testing',
        'drip_track',
        'packaged_app_testing',
        'tricentis_flood',
        'TOUCHPOINT_DATE',
        'EMAIL_ADDRESS__MD5_',
        'n-co-2',
        'final_title',
        'n-co-1',
        'tactical_user',
        'sap_testing',
        'bi/dwh_testing',
        'PERIOD',
        'ACTION',
        'SUBSTANTIVE',
        'ABBR',
        'mobile_testing',
        'third_party_webinar',
        'strategic_quality',
        'bdd',
        'ANN_TOUCHPOINT_ID__MD5_',
        'test_data_management',
        'PREVIOUS_CONTENT',
        'content_syndication',
        'continuous_testing',
        'service_virtualization',
        'n-co-3',
        'open_source_testing',
        'tricentis_rpa',
        'tactical_tester',
        'tricentis_platform',
        'future_of_testing',
        'company_news',
        'tricentis_tosca',
        'business_applications',
        'ID',
        'se-co-3',
        'developer-tester_alignment',
        'customer_success',
        'digital_transformation',
        'CONTENT_OFFER_SERIAL',
        'opportunity_created_date'
    ]
    df_merged.drop(drop_cols, axis=1, inplace=True)

    # remove duplicates
    df_merged.drop_duplicates(inplace=True)

    # filter out all company emails - likely testing accounts
    df_merged = df_merged[~df_merged['EMAIL_ADDRESS'].str.contains('tricentis')]

    # roll up to one row per email
    df_merged = df_merged.groupby('EMAIL_ADDRESS').max()

    return df_merged
