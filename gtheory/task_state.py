'''

Used clustering technique

https://machinelearningmastery.com/clustering-algorithms-with-python/
'''
import os.path

import numpy as np
import pandas as pd


def used_kmeans ():
    b = 1


def traditional_approach ():
    n = 1


def get_global (df_c, swindow,threshold_reaction=9):
    '''
    This function served two purpose
        -Calculate the global reaction time.
        - Filtered events with rt_local greater than  than 9s.
    Parameters
    ----------
    df_c
    swindow: w w centre w w
    threshold_reaction

    Returns
    -------
    Validation: https://stackoverflow.com/a/69269917/6446053
    '''
    df_c["rt_local"].where(df_c["rt_local_sec"] <= threshold_reaction, np.nan, inplace=True)

    df_m = df_c [df_c.event_id.isin ( ['251', '252'] )].reset_index ( drop=True )
    # Old approach that include nan
    # df_m ['rt_global'] = df_m ['rt'].rolling ( 2 * swindow + 1, center=True, min_periods=1 ).mean ()

    m = df_m.isna().any(axis='columns')

    df_m ['rt_global'] = (df_m ["rt_local"].groupby(m.cumsum()).rolling(2 * swindow + 1, center=True, min_periods=1).mean()
                    .reset_index(level=0, drop=True))
    df_m ['rt_global'] = df_m ['rt_global'][~m]

    df_m ['rt_global_sec'] = df_m ['rt_global'] / df_m ['sfreq']


    return df_m


def reaction_time (fpath,pt,**kwargs):
    '''
    Calculate the reaction time Response onset {251 | 252} - Deviation onset {253}
    :param fpath:
    :return:
    '''
    
    # session = kwargs.get('session',None)
    d,f=os.path.split(fpath)
    spath=os.path.join(d,pt["event_log_lgrt"])

    df = pd.read_pickle ( fpath )
    # df.drop(['duration', 'event_key'], axis=1,inplace=True)

    # While it is possible to filter based on the key event_id, but using range
    # as below is more bullet proof
    ro_idx = range ( 0, len ( df ), 3 )
    # Sanity check
    non_desired_event = ['253', '254']
    ro_unique = df.loc [ro_idx, 'event_id'].unique ()
    if np.any ( np.isin ( ro_unique, non_desired_event ) ):
        string = ''.join ( list ( set ( ro_unique.tolist () ).intersection ( non_desired_event ) ) )
        raise ValueError ( f'{string} which should not be here. Please check!' )

    do_idx = range ( 1, len ( df ), 3 )
    do_unique = df.loc [do_idx, 'event_id'].unique ()

    if np.any ( np.isin ( do_unique, ['251', '252', '254'] ) ):
        string = ''.join ( list ( set ( do_unique.tolist () ).intersection ( ['251', '252', '254'] ) ) )
        raise ValueError ( f'{string} which should not be here. PLease check!' )

    df_do = df.loc [do_idx, 'timepoint']
    df_ro = df.loc [ro_idx, 'timepoint']
    if len ( df_ro ) != len ( df_do ): raise ValueError ( f'Uneven length {len ( df_ro )} vs {len ( df_do )} ' )

    df ['event_sec'] = df ['timepoint'] / df ['sfreq']
    df ['rt_local'] = df.timepoint.diff ()
    df ['rt_local'] = df ['rt_local'].shift ( periods=-1 )
    df ['rt_local_sec'] = df ['rt_local'] / df ['sfreq']
    # For time being, we just append new information into this pandas. But, be carefull not
    # change the epoch freq as this will distort the timepoint value.
    df.to_pickle( spath )


def viz_rejected_events (events_all, events_selected,sfolder,rep_xcl):
    '''
    Let do something that we can easily visualize.
    Red cell on the column "result` indicate cell or index_event that we should drop  epochs_event. In other word,
    red cell indicate these values does not exist in df.
    :param events:
    :param df:
    :return:
    '''
    df1=pd.DataFrame(events_all,columns=['id'])
    df2=pd.DataFrame(events_selected,columns=['id'])
    df1 = df1.assign(result=df1['id'].isin(df2['id']).astype(int))
    df1=df1.style.applymap(lambda x: "background-color: red" if x==0 else "background-color: white")
    spaths=os.path.join(sfolder,rep_xcl)
    df1.to_excel(spaths)
    t = 1


def rejected_events (events, df,**kwargs):
    # https://stackoverflow.com/q/50449088/6446053
    events_all = events [:, 0]
    filter_mid= kwargs.get('filter_mid',None)
    path_event_df= kwargs.get('path_event_df',None)
    d,_=os.path.split(path_event_df)
    viz= kwargs.get('path_event_df',None)
    session= kwargs.get('session',None)
    # if session is not None:
    #     # nclus=f'ncluster_5_ss_{session}'
    nclus=f'ncluster_5'
    
    if filter_mid:
        
        ls_val=[0,4]
        df_s = df [df[nclus].isin(ls_val)].reset_index(drop=True)
        df_s.to_pickle(path_event_df)
        events_selected = df_s ['timepoint'].to_numpy ().transpose ()
        rep_xcl=f"visualise_drop_index_04_{session}.xlsx"

    else:
        
        events_selected = df ['timepoint'].to_numpy ().transpose ()
        rep_xcl="visualise_drop_index_inred.xlsx"
    
    if viz:
     
        viz_rejected_events (events_all, events_selected,d,rep_xcl)
    # epochs_drop_idx = np.where ( ~np.in1d ( events_all, events_selected ) ) [0]
    return np.where ( ~np.in1d ( events_all, events_selected ) ) [0]

