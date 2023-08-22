import json
import os
import shutil
from os.path import (split)
from re import findall, search, split

import numpy as np
import pandas as pd

def filter_path(fpath,**kwargs):
    dsession=kwargs['session'].split('_')
    if 'case' in dsession:
        """
        If case is in side, do some
        """
        
        if kwargs['session'] =='n_case_A':
            ls_sbj=['060926n1','060926n2','060720n1','090825n',
                '070124n','061225n','080530n','070307n']
            # ls_sbj=['060926n2','060720n1','090825n',
            #             '070124n','061225n','080530n','070307n']
    case=dict()
    dir,fname=os.path.split(fpath)
    next=fname.split('.')[-1]
    # sbj = findall ( r's\d{2}', fpath, flags=re.IGNORECASE ) [0].upper ()
    sub_session = findall ( r'\d{6}', fpath ) [0]
    # hs=fpath.split(sub_session)[-1].split(fname)[0].split(os.sep)
    h=fpath.split(sub_session)[-1].split(fname)[0]
    
    if 'case' in dsession:
        
        path = os.path.normpath(dir)
        dcomp=path.split(os.sep)
        fff=set(ls_sbj).isdisjoint(set(dcomp))
        if not fff:
            return fpath

    if 'case' not in dsession and kwargs['session'] in h:
        return fpath
    
def get_omega_gamma(fpath,algo_experiment):
    """
    algo_experiment: Type of experiment
    'louvain'

    fpath: 'gamma_omega.xlsx'
    """

    from collections import defaultdict

    df = pd.read_excel(fpath, index_col=0, sheet_name=algo_experiment).reset_index()
    df['dict_key'] = df[['fband', 'method']].agg('_'.join, axis=1)

    df = df.drop(columns=['comment', 'fband', 'method'])
    # df['gamma'] = df['gamma'].astype('int64')
    # df['omega'] = df['omega'].astype('int64')

    adic=df.to_dict('records')
    omega_gamma_pair = defaultdict(dict)

    for person in adic:
        omega_gamma_pair[person['dict_key']]['omega'] = person['omega']
        omega_gamma_pair[person['dict_key']]['gamma'] = person['gamma']

    # omega_gamma_pair = [{x['dict_key']: {"omega": x['omega'], "gamma": x['gamma']}} for x in adic]

    return omega_gamma_pair


def drop_overlap_event(fpath, pt, overlap_threshold=7):
    '''
    For this study, overlap event will be drop.
    epoch_event
    2,            nan
    13,           11.00000
    15,           2.00000
    25,           10.00000
    36,           11.00000
    40,           4.00000
    45,           5.00000

    Retain only

    2,            nan
    13,           11.00000
    25,           10.00000
    36,           11.00000


    Example
    overlap_threshold=9  # unit is in second
    df=pd.DataFrame([2,13,15,25,36,40,45],columns=['epoch_event'])
    df['previous_time_diffrence']=df.epoch_event.diff()
    df=df[(df['previous_time_diffrence']>overlap_threshold)|(df.previous_time_diffrence.isnull())]
    '''

    dpath, _ = os.path.split(fpath)
    spath = os.path.join(dpath, pt['filtered_event_fn'])
    df = pd.read_pickle(fpath)
    # df = pd.read_feather(fpath)
    df['previous_time_diffrence'] = df.event_sec.diff()
    df = df[
        (df['previous_time_diffrence'] > overlap_threshold) | (df.previous_time_diffrence.isnull())]

    # """
    # The following lines serve two purpose
    # 1) Avoid drop_log with title "NO_DATA" which occur for data:S01_060926n. For pt['tmin_do'] and pt['fsampling']
    # equavalent to -8 and 256 Hz, the first event label as "NO_DATA"
    #
    # 2) This allow us to define after how many second we allow to pick an event.
    # """
    shift_start_event = 60  # Unit in Sec.means we start using event 60S after the experiment start

    df = df[df['event_sec'] > shift_start_event]

    ## In some instance, the rt_local and rt_global have value of nan
    df = df.dropna().reset_index(drop=True)

    # Create a new file with the name 'filtered-event.feather'
    # df.to_feather(spath)
    df.to_pickle(spath)


def node_subsystem(d):
    '''


      :param d: Label each node to its own subsustem from a lookup dictionary
      "brain_region_id": {
          "frontal": [0,1,3,4,5],
          "temporalleft": [2,7,12,17,22],
          "temporalright": [6,11,16,21,26],
          "central": [8,9,10,13,14,15,18,19,20],
      "occipital":[23,24,25,27,28,29]

      :return: susbsytem for each node

      See: https://stackoverflow.com/a/70048082/6446053
      '''
    inv_m = {v: k for k, lst in d.items() for v in lst}
    _, opt = np.unique(np.array([inv_m[v] for v in range(len(inv_m))]), return_inverse=True)
    # out=[inv_m[v] for v in range(len(inv_m))]
    return opt


def load_json(fname):
    with open(fname, 'r') as f:
        return json.load(f)


def extract_motion_normal(s):
    exp_set = search(r'\d{6}(.*?).set', s).group(1)
    if len(findall('\d+', exp_set)) > 0:
        exp_set = split("[^a-zA-Z]", exp_set)[0]
    return exp_set


# def split_session(fpath) :
#     root_folder,fname=split(fpath)
#     sub_session=findall(r'\d+',fname)[0]
#     return sub_session,root_folder


def split_id_session(fpath):
    '''
      :param fpath:
      :return:
      '''

    # root_folder,fname=split(fpath)
    root_folder = split(fpath)
    match = search(r'(\w+)_(\w+)[^/]*$', fpath)
    sbj, sub_session = match.group(1).upper(), match.group(2)
    nlen = len(list(sub_session))
    if nlen < 3:
        import re
        sbj = findall(r's\d{2}', fpath, flags=re.IGNORECASE)[0].upper()
        sub_session = findall(r'\d{6}', fpath)[0]
        session_repeat = re.search(fr'{sub_session}_(.*?).set', fpath).group(1)
        session_repeat = list(session_repeat)
        session_repeat.reverse()
        sub_session = sub_session + "".join(session_repeat)

    return sbj, sub_session, root_folder


def sort_path(path_all):
    import re
    path_all.sort(key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])
    return path_all


def check_make_folder(path, remove=False):
    if not remove:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
    else:
        '''
        Some time I want the folder to be emptied
        '''
        if os.path.exists(path):
            shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)
        else:
            os.makedirs(path, exist_ok=True)

# def load_file(fname,warning_level=None) :
#     import pickle
#     if not os.path.isfile(fname) :
#         if warning_level is not None :
#             raise ValueError('Try to upload something that not there')
#
#     with open(fname,'rb') as handle :
#         file=pickle.load(handle)
#     return file

#
# def easy_sort_idx(wind_name,window_size) :
#     import numpy as np
#     '''
#       Extract the 2D coordinate only for window with the 0s on it. This usually helpfull to sort the figure
#       '''
#     # window_size, txt = 7,['-25:-20', '-20:-15', '-15:10', '-10:-5', '-5:0', '0:5', '5:10']
#     ref_val=[i for i in range(len(wind_name)) if '0' in wind_name[i].split(":")]
#     bt,lf=np.tril_indices(window_size,-1)
#     pos_window=0
#
#     indices=np.where(bt==ref_val[pos_window])
#     idx_pair_for_easy_sorting=[bt[indices],lf[indices]]
#     return idx_pair_for_easy_sorting

#
# def get_sub_folder(path) :
#     return [f.path for f in os.scandir(path) if f.is_dir()]
