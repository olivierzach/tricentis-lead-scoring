from data.data_util import *

# TODO: which leads should we filter to? prospect, customer, unknown?


def profile_leads_data(df):

    # filter out the different lead types - only want wins, prospects, unknown
    keep_account_types = ['Prospect', 'Customer', 'Unknown']
    df = df[df.account_type.isin(keep_account_types)]

    # group countries by distribution of appearance on leads
    keep_countries = [
        'united states',
        'india',
        'united kingdom',
        'australia',
        'germany',
        'canada',
        'netherlands',
        'austria',
        'switzerland',
        'singapore',
        'sweden',
        'philippines',
        'france' ,
        'brazil',
        'poland',
        'new zealand',
        'mexico',
        'china'
    ]

    # flag if lead in top countries
    df['country_flag_common'] = np.where(
        df.country.str.lower().isin(keep_countries),
        1,
        0
    )

    # explicit country missing flag
    df['country_flag_missing'] = np.where(
        df.country.str.lower().isna(),
        1,
        0
    )

    # individual flags for the most common countries
    for i in keep_countries:
        name = i.replace(" ", "_")

        df["country_flag_" + name] = np.where(
            df.country.str.lower().isin([i]),
            1,
            0
        )

    # group states by distribution of appearance on leads
    keep_states = [
        'california',
        'texas',
        'new york',
        'illinois',
        'georgia',
        'massachusetts',
        'new jersey',
        'ohio',
        'florida',
        'north carolina',
        'new south wales',
        'virginia',
        'pennsylvania',
        'washington',
        'minnesota',
        'victoria',
        'michigan',
        'colorado',
        'wisconsin',
        'connecticut',
        'maryland',
        'arizona',
        'missouri',
        'tennessee',
        'ontario',
        'indiana',
        'new mexico',
        'utah'
    ]

    # flag if lead in top states
    df['state_flag_common'] = np.where(
        df.state.str.lower().isin(keep_states),
        1,
        0
    )

    # explicit state missing flag
    df['state_flag_missing'] = np.where(
        df.state.str.lower().isna(),
        1,
        0
    )

    # individual flags for the most common states
    for i in keep_states:
        name = i.replace(" ", "_")

        df["state_flag_" + name] = np.where(
            df.state.str.lower().isin([i]),
            1,
            0
        )

    # group conversation tracks by distribution of appearance on leads
    keep_tracks = [
        'strategic it',
        'business transformation',
        'unclassified',
        'strategic quality',
        'business applications'
    ]

    # flag if lead in the top conversation tracks
    df['conversation_track_flag_common'] = np.where(
        df.conversation_track.str.lower().isin(keep_tracks),
        1,
        0
    )

    # explicit conversation track missing flag
    df['conversation_track_flag_missing'] = np.where(
        df.conversation_track.str.lower().isna(),
        1,
        0
    )

    # individual flags for the most common conversation tracks
    for i in keep_tracks:
        name = i.replace(" ", "_")

        df["conversation_track_flag_" + name] = np.where(
            df.conversation_track.str.lower().isin([i]),
            1,
            0
        )

    # group the test management solution question
    keep_test_management = [
        'spreadsheets',
        'quality center',
        'zephyr for jira add-on',
        'xray for jira add-on'
    ]

    # flag if in common products
    df['test_management_flag_common'] = np.where(
        df.test_management_solution.str.lower().isin(keep_test_management),
        1,
        0
    )

    # explicit test management missing flag
    df['test_management_flag_missing'] = np.where(
        df.test_management_solution.str.lower().isna(),
        1,
        0
    )

    # individual flags for the most common test management answers
    for i in keep_test_management:
        name = i.replace(" ", "_").replace("'", "").replace("-", "")

        df["test_management_flag_" + name] = np.where(
            df.test_management_solution.str.lower().isin([i]),
            1,
            0
        )

    # group the time frame question
    keep_time = [
        'currently evaluating solutions',
        '1-3 months',
        '3-6 months',
        'more than 12 months',
        '6-12 months'
    ]

    # flag if in short term time frame
    df['time_frame_flag_short_term'] = np.where(
        df.trimeframe.str.lower().isin(keep_time[:2]),
        1,
        0
    )

    # flag if in long term time frame
    df['time_frame_flag_long_term'] = np.where(
        df.trimeframe.str.lower().isin(keep_time[2:]),
        1,
        0
    )

    # explicit time frame missing flag
    df['time_frame_flag_missing'] = np.where(
        df.trimeframe.str.lower().isna(),
        1,
        0
    )

    # flag if active project in current plan
    df['active_project_flag_current'] = np.where(
        df.active_project.str.lower().replace(",", "").isin(['yes currently']),
        1,
        0
    )

    # flag if active project in short term
    df['active_project_flag_short_term'] = np.where(
        df.active_project.str.lower().replace(",", "").isin(['yes currently', 'no but will be soon']),
        1,
        0
    )

    # flag if active project in long term
    df['active_project_flag_long_term'] = np.where(
        df.active_project.str.lower().replace(",", "").isin(['dont know', 'no plans to evaluate']),
        1,
        0
    )

    # group the primary interest column
    keep_interest = [
        'test automation',
        'test management and reporting',
        'test automation - sap technology stack',
        'robotic process automation (rpa)',
        'bdd/tdd',
        'open source test automation management'
    ]

    # common interests
    df['primary_interest_flag_common'] = np.where(
        df.primary_area_of_interest.str.lower().isin(keep_interest),
        1,
        0
    )

    # explicit missing flag
    df['primary_interest_flag_missing'] = np.where(
        df.primary_area_of_interest.str.lower().isna(),
        1,
        0
    )

    # develop % missing on form feature
    form_cols = [
        'test_management_solution',
        'current_state_alm',
        'future_state_alm',
        'current_state_defect_tracking',
        'future_state_defect_tracking',
        'trimeframe',
        'active_project',
        'country',
        'state',
        'primary_area_of_interest'
    ]

    # convert unsubscribe to binary
    df['unsubscribed_from_email_flag'] = df['unsubscribed_from_email_flag'].astype(int)

    # how many questions asked on entire form were not completed?
    df['form_fill_missing_pct'] = df[form_cols].isna().sum(axis=1) / len(form_cols)

    # check if email is valid
    df['valid_email_flag'] = df['email'].apply(lambda x: is_valid_email(x)).astype(int)

    # parse out email for business information
    df = url_parsing_business(df, url_field='email_domain')

    # dummy out the email and email domain columns
    cols_to_dummy = ['email_domain']
    df = dummy_wrapper(df, cols_to_dummy=cols_to_dummy)

    # convert date columns
    date_cols = [c for c in df.columns if 'date' in c]
    for i in date_cols:
        df[i] = pd.to_datetime(df[i], format='%Y-%m-%d', errors='ignore')

    # columns we can drop - need only leads data - opportunity information will leak information
    # also cannot use current scores - need score at time lead sent to floor
    # also drop all base feature columns
    drop_cols = [
        'country',
        'state',
        'account_type',
        'conversation_track',
        'test_management_solution',
        'current_state_alm',
        'future_state_alm',
        'current_state_defect_tracking',
        'future_state_defect_tracking',
        'trimeframe',
        'active_project',
        'role',
        'lead_account_id',
        'lead_title',
        'current_total_score',
        'current_demographic_score',
        'current_behavior_score',
        'primary_area_of_interest',
        'passed_to_sales',
        'accepted_by_sales',
        'account_opportunity_id',
        'opportunity_type',
        'opportunity_sub_type',
        'opportunity_products',
        'opportunity_subscription_term_in_months',
        'opportunity_arr_usd',
        'opportunity_tcv_usd',
        'opportunity_status',
        'lost_reason',
        'initiative_started',
        'competitors_involved',
        'bookings_team',
        'sales_rep_id',
        'applications_being_tested',
        'opportunity_set_with'

    ]
    df.drop(drop_cols, axis=1, inplace=True)

    # set index of meta data and keys
    idx_cols = [
        'email',
        'lead_id',
        'opportunity_id',
        'opportunity_created_date',
        'opportunity_close_date',
        'sales_accepted_date',
        'opportunity_won'
    ]
    df.set_index(idx_cols, inplace=True)

    return df

