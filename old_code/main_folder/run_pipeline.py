'''
Sequence of pipeline

extract_delete_zip ()
load_to_epoch (debug=False)
cal_reaction_time (debug=False)
get_reaction_time_global ()
remove_overlap_events()
filter_epoch_to_fulfiled_condition (debug=False)
get_cluster_model ()
cluster_assignment ()
get_cluster_per_subject_report ()
combine_all_epochs_for_csd()
make_csdx (debug=True)
epoch_to_epochs (debug=True)
make_connectivity (debug=True)
make_gtheory()
cal_flexibility ()
assign_plot_flexibility()
plot_flexibility()


'''
import logging
import os.path
import sys
from datetime import datetime

# https://stackoverflow.com/a/11548754/6446053 for logger
logging.getLogger().setLevel(logging.INFO)
# logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
# logging.getLogger().setLevel(logging.DEBUG)
# logging.debug('This is a debug message')
# logging.info('This is an info message')
# logging.warning('This is a warning message')
# logging.error('This is an error message')
# logging.critical('This is a critical message')
# k=1

gettrace = sys.gettrace()
now = datetime.now()
# For debugging
debug_status = True if gettrace else False
import re
from re import findall
from old_code.gtheory.utils.misc import filter_path
from tqdm import tqdm


def extract_delete_zip():
    from old_code.gtheory.utils.misc import sort_path
    from old_code.gtheory import config as cf
    from glob import glob
    from old_code.gtheory import extract_remove_zip
    pt = cf.root_path()
    path_all = sort_path(glob(f"{pt['dir_root']}/*/*.zip"))
    
    extract_remove_zip(path_all, pt['dir_root'], remove_zip=False)
    t = 1


        
    
  
def load_to_epoch(**kwargs):
    from old_code.gtheory import (load_raw_to_epochs)
    from old_code.gtheory.utils.misc import sort_path
    from glob import glob

    from old_code.gtheory import config as cf

    # kwargs['debug']=True
    pt = cf.root_path()
    psbjs_all = sort_path(glob(f"{pt['dir_root']}/*/*/*.set"))

    if kwargs['session'] is not None:
        psbjs = list(filter(None.__ne__, [filter_path(fpath,**kwargs) for fpath in psbjs_all]))
    else:
        psbjs=psbjs_all
    
    if kwargs['debug']:
        sbj1_path = psbjs[0]
        load_raw_to_epochs(sbj1_path, pt)
    else:
        

        logging.info(f'Process: load_to_epoch for total {len(psbjs)} subjects in parallel mode')
        [load_raw_to_epochs ( sbj,pt,debug=False) for sbj in psbjs ]
        # Parallel(n_jobs=-1, verbose=50)(
        #     delayed(load_raw_to_epochs)(sbj, pt, debug=False) for sbj in psbjs)


def cal_reaction_time(**kwargs):
    from old_code.gtheory import reaction_time
    from old_code.gtheory.utils.misc import sort_path
    from joblib import Parallel, delayed
    from glob import glob
    from tqdm import tqdm
    from old_code.gtheory import config as cf
    pt = cf.root_path(**kwargs)
    njob = 1
    psbjs_all = sort_path(glob(f"{pt['dir_root']}/*/*/{pt['raw_event_fn']}"))
    

    
    if kwargs['session'] is not None:
        psbjs = list(filter(None.__ne__, [filter_path(fpath,**kwargs) for fpath in psbjs_all]))
    else:
        psbjs=psbjs_all
    
    if kwargs['debug']:
        reaction_time(psbjs[0])
    else:
        
        if njob == 1:
            logging.info(
                f'Process: Calculate local reaction time for a total  {len(psbjs)} subjects in single cpu mood')
            [reaction_time(my_path,pt,**kwargs) for my_path in tqdm(psbjs)]
        
        else:
            logging.info(
                f'Process: Calculate local reaction time for a total  {len(psbjs)} subjects in parallel mood')
            Parallel(n_jobs=-1, verbose=50)(
                delayed(reaction_time)(my_path,pt,**kwargs) for my_path in tqdm(psbjs))


def get_reaction_time_global(**kwargs):
    from glob import glob
    from old_code.gtheory import get_global as gg
    from old_code.gtheory.utils.misc import sort_path
    import pandas as pd
    from tqdm import tqdm
    from old_code.gtheory import config as cf

    pt = cf.root_path(**kwargs)
    swindow = 2
    

    psbjs_all = sort_path(glob(f"{pt['dir_root']}/*/*/{pt['event_log_lgrt']}"))
    
    if kwargs['session'] is not None:
        psbjs = list(filter(None.__ne__, [filter_path(fpath,**kwargs) for fpath in psbjs_all]))
    else:
        psbjs=psbjs_all
        

    
    logging.info(f'Process: Calculate global reaction time for a total  {len(psbjs)} subjects')
    for fpath in tqdm(psbjs):
        df = gg(pd.read_pickle(fpath), swindow)
        # Overwrite the old file with these latest rt_global information
        df.to_pickle(fpath)


def remove_overlap_events(**kwargs):
    '''
    I would like to remove events that overlaps to each other.
    '''
    from glob import glob
    from old_code.gtheory.utils.misc import sort_path, drop_overlap_event
    from tqdm import tqdm
    from old_code.gtheory import config as cf
    pt = cf.root_path(**kwargs)
    overlap_threshold = pt['overlap_threshold']  # Unit is in seconds
    
    psbjs_all = sort_path(glob(f"{pt['dir_root']}/*/*/{pt['event_log_lgrt']}"))
    psbjs = list(filter(None.__ne__, [filter_path(fpath,**kwargs) for fpath in psbjs_all]))
    logging.info(f'Process: Remove overlap events for a total  {len(psbjs)} subjects')
    for fpath in tqdm(psbjs):
        drop_overlap_event(fpath, pt, overlap_threshold=overlap_threshold)

def njoin_new_fname(dpath,nfame):
    d,_=os.path.split(dpath)
    npath=os.path.join(d,nfame)
    return npath

def filter_epoch_to_fulfiled_condition(**kwargs):
    '''
    When first import events from epoch, all event 231,232,233,234 were included in the <raw-event.feather>.
    For rt_local calculation for each deviation onset event,we need the full cycle id(251,252,253,254). Apart from
    rt_local calculation, the id 253 and 254 serve no purpose in the subsequent analysis. Hence, at this stage,
    we need to remove this orphan events and mark the process as 'USER REASON'.

    Hence, since we are dropping some events based on certain condition, we rename the file as filtered-event.feather).

    '''
    from glob import glob
    from old_code.gtheory import config as cf
    from tqdm import tqdm
    from joblib import Parallel, delayed
    from old_code.gtheory.utils.misc import sort_path
    from old_code.gtheory import drop_epoch_based_condition
    pt = cf.root_path(**kwargs)
    psbjs_all = sort_path(glob(f"{pt['dir_root']}/*/*/{pt['prep_epo_fn']}"))
    
    if kwargs['session'] is not None:
        path_epoch = list(filter(None.__ne__, [filter_path(fpath,**kwargs) for fpath in psbjs_all]))
    else:
        path_epoch=psbjs_all
        
    
    
    fpath=[njoin_new_fname(dpath,pt['kmean_c_fn']) for dpath in path_epoch]
    njob = 1
    kwargs['debug']=False
    if kwargs['debug']:
        # opath=['/mnt/d/data_set/sustained_attention_driving/S01/060227n/community_leiden/epoch_cp/coh_theta_epoch_cp.npz']
        logging.info(f'Debug Mood for the CSD at {now.strftime("%d_%m_%Y_%H_%M_%S")}')
        # path_epoch = path_epoch[0]
        # fpath = sort_path(glob(f"{pt['dir_root']}/*/*/{pt['filtered_event_fn']}"))[0]
        drop_epoch_based_condition(path_epoch[0], fpath[0], pt,**kwargs)
    else:
        # path_epoch = sort_path(glob(f"{pt['dir_root']}/*/*/{pt['prep_epo_fn']}"))
        
        # fpath = sort_path(glob(f"{pt['dir_root']}/*/*/{pt['filtered_event_fn']}"))
        if njob != 1:
            logging.info(
                f'Process: Remove overlap events for a total  {len(fpath)} subjects in parallel mode')
            Parallel(n_jobs=-1, verbose=50)(
                delayed(drop_epoch_based_condition)(path_epoch_, fpath_, pt,**kwargs) for path_epoch_, fpath_
                    in zip(path_epoch,
                    fpath))
        else:
            logging.info(
                f'Process: Remove overlap events for a total  {len(fpath)} subjects in single cpu mode')
            [drop_epoch_based_condition(path_epoch_, fpath_, pt,**kwargs) for path_epoch_, fpath_ in
                tqdm(zip(path_epoch, fpath), desc='Filter only event')]

def get_cluster_model(**kwargs):
    from glob import glob
    from old_code.gtheory import config as cf
    from old_code.gtheory import fit_kmeans
    from old_code.gtheory.utils.misc import sort_path
    from tqdm import tqdm
    import pandas as pd
    
    pt = cf.root_path(**kwargs)
    psbjs_all = sort_path(glob(f"{pt['dir_root']}/*/*/{pt['filtered_event_fn']}"))
    fpath = list(filter(None.__ne__, [filter_path(fpath,**kwargs) for fpath in psbjs_all]))
    logging.info(f'Process: Get Cluster model using data from a total  {len(fpath)} dataset '
                 f'in single cpu mode')
    # Read rt_local and rt_global for all subject and store as a list. In the <fit_kmeans>, the files will be concat
    # as a single df
    df_alist = [pd.read_pickle(fpath) for fpath in tqdm(fpath, desc='Read df for each subject')]
    nclus = [2, 3, 4, 5]
    kwargs['psave']=pt['kmean_model']
    for ncluster in tqdm(nclus):
        fit_kmeans(df_alist, ncluster,**kwargs)


def cluster_assignment(**kwargs):
    '''
    Define class identity for each events (i.e.,each deviation onset) based on the kmeans model. The result will
    be logged and saved into new df file with the name `filtered-event.feather`
    '''
    
    from glob import glob
    from old_code.gtheory import config as cf
    from tqdm import tqdm
    from old_code.gtheory import predict_kmeans_model
    from old_code.gtheory.utils.misc import sort_path
    import pandas as pd
    pt = cf.root_path(**kwargs)
    
    psbjs_all = sort_path(glob(f"{pt['dir_root']}/*/*/{pt['filtered_event_fn']}"))

    psbjs = list(filter(None.__ne__, [filter_path(fpath,**kwargs) for fpath in psbjs_all]))
    session = kwargs.get('session', None)
    
    if  session is None:
        fext = f'.pickle'
    else:
        fext = f'_ss_{session}.pickle'
    logging.info(
        f'Process: Get event classification for each data from a total  {len(psbjs)} dataset '
        f'in single cpu mode')
    for fpath in tqdm(psbjs):
        root_folder, _ = os.path.split(fpath)
        df = pd.read_pickle(fpath)
        pname, fname = os.path.split(fpath)
        spath = os.path.join(pname, pt['kmean_c_fn'])
        k, sbj_session = os.path.split(pname)
        sbj_id = os.path.basename(k)
        sbj_id_session = sbj_id + '_' + sbj_session
        for nclus in [2, 3, 4, 5]:
            
            fkmean = os.path.join(pt['kmean_model'],f'kmean_cluster_{nclus}{fext}' )
            predict_kmean = predict_kmeans_model(fkmean)
            df = predict_kmean.predict_cluster(df, sbj_id=sbj_id_session)
        
        df.to_pickle(spath)


def get_cluster_per_subject_report(**kwargs):
    from glob import glob
    from old_code.gtheory import config as cf
    from old_code.gtheory.utils.misc import sort_path
    from old_code.gtheory.utils import save_list_df_multiple_sheets
    from old_code.gtheory.reporting import get_cluster_per_subject
    pt = cf.root_path(**kwargs)
    session= kwargs.get('session',None)
    dstring=session
    cluster_cl = [2, 3, 4, 5]
    dcase=[f'ncluster_{idx}' for idx in cluster_cl]

    psbjs = sort_path(glob(f"{pt['dir_root']}/*/*/{pt['kmean_c_fn']}"))
    all_df = [get_cluster_per_subject(psbjs, nclus) for nclus in dcase]
    
    from old_code.gtheory.utils.misc import check_make_folder
    
    check_make_folder(pt['report'])
    spath = os.path.join(pt['report'], pt['cluster_kmeans_distribution_fn'])
    save_list_df_multiple_sheets(all_df, cluster_cl, spath)


def combine_all_epochs_for_csd():
    '''

      - Objective to concat all subject event is to have an overview about optimum csd parameter selection
      :return:
      '''
    from mne import read_epochs, concatenate_epochs
    from old_code.gtheory.utils.misc import check_make_folder
    from glob import glob
    from old_code.gtheory import config as cf
    from old_code.gtheory.utils.misc import sort_path
    from tqdm import tqdm
    import random
    pt = cf.root_path()
    check_make_folder(pt['source_local'])
    spath = os.path.join(pt['source_local'], pt['epochs_concat_for_csd_evaluation'])
    path_epoch = sort_path(glob(f"{pt['dir_root']}/*/*/*filtered-epo.fif"))
    epochs = [read_epochs(path_ep, preload=False) for path_ep in tqdm(path_epoch)]
    all_epoch = []
    nevents_to_use = 50  # 70 epochs will result in FIF of size 1G. 60: 893MB,50:744Mb
    for eepoch in tqdm(epochs):
        epo_idx_available = eepoch.selection.tolist()
        random.shuffle(epo_idx_available)  # In place exercise, no return value
        idx_retain = epo_idx_available[:nevents_to_use]
        epochs_drop_idx = list(set(range(len(eepoch))) - set(idx_retain))
        eepoch.drop(epochs_drop_idx, reason='User reason')
        all_epoch.append(eepoch)
    
    epochs_standard = concatenate_epochs(all_epoch)
    epochs_standard.save(spath, overwrite=True)
    b = 1


def csd_parameter_evaluation_selection():
    '''
    KIV
    '''
    from mne import read_epochs
    from old_code.gtheory import config as cf
    pt = cf.root_path()
    opath = os.path.join(pt['source_local'], pt['epochs_concat_for_csd_evaluation'])
    epoch = read_epochs(opath, preload=False)
    v = 1


def make_csdx(**kwargs):
    from joblib import Parallel, delayed
    from glob import glob
    from old_code.gtheory import config as cf
    from old_code.gtheory.utils.misc import sort_path
    from tqdm import tqdm
    from old_code.gtheory import (make_csd)
    pt = cf.root_path(**kwargs)
    njob = 1
    

    k=pt['filtered_epo_fn']
    fpath = sort_path(glob(f"{pt['dir_root']}/*/*/{k}"))
    
    
    if kwargs['debug']:
        # opath=['/mnt/d/data_set/sustained_attention_driving/S01/060227n/community_leiden/epoch_cp/coh_theta_epoch_cp.npz']
        # Used default value  lambda2=1e-5, stiffness=4, n_legendre_terms=50
        logging.info(f'Debug Mood for the CSD at {now.strftime("%d_%m_%Y_%H_%M_%S")}')

        # make_csd(sbj1_path,pt)
        [make_csd(sbj1_path, pt,**kwargs) for sbj1_path in
            tqdm(fpath, desc='Run the csd using default value')]
    
    else:
        
        if njob != 1:
            logging.info(f'Parallel Mood for the CSD at {now.strftime("%d_%m_%Y_%H_%M_%S")}')
            Parallel(n_jobs=-1, verbose=50)(delayed(make_csd)(sbj1_path, pt,**kwargs) for sbj1_path in fpath)
        else:
            logging.info(f'Single CPU  Mood for the CSD at {now.strftime("%d_%m_%Y_%H_%M_%S")}')
            [make_csd(sbj1_path, pt,**kwargs) for sbj1_path in
                tqdm(fpath, desc='Run the csd using default value')]


def _epoch_to_epochs(fband_window_max,**kwargs):
    '''
    I have visually check, the figure epoch and epochs_slices is tally. More importantly, the first epoch_slice is
    window far away from the deviation onset, and the last epoch_slice is the closest to the deviation onset. The
    snipet for the verification available in _mne.py of the function sanity_check_original_epoch_to_slicesepochs()
    '''
    from joblib import Parallel, delayed
    from glob import glob
    from old_code.gtheory import config as cf
    from old_code.gtheory.utils.misc import sort_path
    from tqdm import tqdm
    from old_code.gtheory import (sliding_epoch_to_epochs)
    kwargs['fband']=fband_window_max
    pt = cf.root_path(**kwargs)
    overlap_window=pt['overlap_window'] # For time being, always set to 0 (non-overlap)
    window_epoch_size = cf.fband_con_max[fband_window_max]['window_epoch_size']
    njob = 1

    file_ext=pt['csd_epo_fn']

    fpath = sort_path(glob(f"{pt['dir_root']}/*/*/{file_ext}"))
    kwargs['debug']=False
    if kwargs['debug']:
        logging.info(f'Debug Mood for the epoch to epochs at {now.strftime("%d_%m_%Y_%H_%M_%S")}')
        
        sliding_epoch_to_epochs(fpath[0], fband_window_max,pt, overlap_window=overlap_window,
            window_epoch_size=window_epoch_size,
            return_ep=False,**kwargs)
    else:
        
        if njob != 1:
            logging.info(
                f'Parallel Mood for the epoch to epochs at {now.strftime("%d_%m_%Y_%H_%M_%S")}')
            Parallel(n_jobs=-1, verbose=50)(delayed(sliding_epoch_to_epochs)(sbj1_path,pt,
                window_epoch_size=window_epoch_size,
                overlap_window=overlap_window,**kwargs)
                for sbj1_path in fpath)
        else:
            logging.info(
                f'Single CPU Mood for the epoch to epochs at {now.strftime("%d_%m_%Y_%H_%M_%S")}')
            [sliding_epoch_to_epochs(sbj1_path, pt,
                                    window_epoch_size=window_epoch_size,
                                    overlap_window=overlap_window,**kwargs) for sbj1_path in
                                    tqdm(fpath, desc='Generate epochs from single epoch')]


def _make_connectivity(fband_window_max,**kwargs):
    '''
    KIV: SEBAB RHYS KACAU
    '''
    from joblib import Parallel, delayed
    from glob import glob
    from old_code.gtheory import config as cf
    from old_code.gtheory.utils.misc import sort_path
    from tqdm import tqdm
    from old_code.gtheory import (get_spectral_connectivity)
    kwargs['fband']=fband_window_max
    pt = cf.root_path(**kwargs)
    njob = 1
    fband_dict = cf.fband_con_max[kwargs['fband']]['freq_bands']
    HH=pt['overlapped_epo_fn']

    fpath = sort_path(glob(f"{pt['dir_root']}/*/*/*/{HH}"))
    # debug=True
    kwargs['debug']=False
    if kwargs['debug']:
        logging.info(
            f'Debug Mood for the connectivity epochs at {now.strftime("%d_%m_%Y_%H_%M_%S")}')
        # sbj1_path = sort_path(glob(f"{pt['dir_root']}/*/*/*/{file_ext}"))[0]
        get_spectral_connectivity(fpath[0], fband_dict,fband_window_max,pt, bsave=True,filter_mid=filter_mid)
    else:
        
        if njob != 1:
            # [Parallel(n_jobs=3)]: Done  62 out of  62 | elapsed: 15.6min finished
            logging.info(
                f'Paralell Mood for the connectivity epochs at {now.strftime("%d_%m_%Y_%H_%M_%S")}')
            Parallel(n_jobs=-1, verbose=50)(
                delayed(get_spectral_connectivity)(sbj1_path, fband_dict,fband_window_max,pt, bsave=True,
                    **kwargs) for sbj1_path in
                    fpath)
        else:
            logging.info(
                f'Single CPU Mood for the connectivity epochs at {now.strftime("%d_%m_%Y_%H_%M_%S")}')
            [get_spectral_connectivity(sbj1_path, fband_dict,fband_window_max,pt, bsave=True,
                    **kwargs) for sbj1_path in
                tqdm(fpath, desc='Filter only event')]
    
    h = 1


def make_gtheory():
    from glob import glob
    from old_code.gtheory import config as cf
    from old_code.gtheory.utils.misc import sort_path
    import numpy as np
    from old_code.gtheory import gt_leidenalg
    pt = cf.root_path()
    
    sbj1_path = sort_path(glob(f"{pt['dir_root']}/*/*/{pt['connectivity_result_fn']}"))[0]
    # sbj1_path = sort_path ( glob ( f"{pt ['dir_root']}/*/*/connectivity_result.npz" ) )
    arr = np.load(sbj1_path, allow_pickle=True)
    con = arr['con_arr']  # (epochs, nwindows, spectral_method, nchan, nchan, fbands)
    params = arr['param'].tolist()
    ch_names = arr['ch_name'].tolist()
    
    # EXTRACT SOME
    filter_indices = [1, 2]
    axis = 0
    BBB = np.take(con, filter_indices, axis)
    # %%
    epoch_idx = 0
    spec_method_idx = 0
    fband_idx = 0
    arr_temporal = con[epoch_idx, :, spec_method_idx, :, :, fband_idx]
    arr_temporal = np.where(arr_temporal < 0.8, 0, arr_temporal)
    from old_code.gtheory.utils import between_list_diff_index
    from old_code.gtheory.utils.nparray import slice_array_with_list
    ch_retain = ['F3', 'FZ', 'F4', 'F8', 'FC3', 'FCZ', 'FC4', 'FT8', 'O1', 'OZ', 'O2']
    retain_idx = between_list_diff_index(ch_names, ch_retain, output='retain_index')
    arr_temporal = slice_array_with_list(arr_temporal, retain_idx.tolist())
    ch_names = ch_retain
    
    nslice = arr_temporal.shape[0]
    slice_name = [f'Slice {idx}' for idx in range(nslice)]
    gtl = gt_leidenalg(arr_temporal, ch_names, slice_name=slice_name)
    gtl.viz_membership_temporal()
    communities = gtl.arr_to_membership()
    communities = communities.values
    
    # https://teneto.readthedocs.io/en/latest/api/teneto.plot.slice_plot.html#teneto.plot.slice_plot
    import teneto
    g_arr = arr_temporal.T
    import matplotlib.pyplot as plt
    tnet = teneto.TemporalNetwork(from_array=g_arr, nodelabels=ch_names)
    tnet.plot('slice_plot', **{'communities': communities}, nodesize=15)
    
    plt.tight_layout()
    plt.show()
    n = 1


def cal_flexibility():
    '''
    I have not run or validate this function as of 02 December 2021

    '''
    # import pickle
    # import numpy as np
    # from gtheory.grapht import gt_leidenalg
    # from gtheory.utils import between_list_diff_index
    # from gtheory.utils.nparray import slice_array_with_list
    from old_code.gtheory import get_community_each_epoch
    sbj1_path = sort_path(glob(f"{pt['dir_root']}/*/*/connectivity_result.npz"))
    for mpath in tqdm(sbj1_path):
        get_community_each_epoch(mpath)


def assign_plot_flexibility():
    '''
    I have not run or validate this function as of 02 December 2021

    '''
    import pickle
    import numpy as np
    from tqdm import tqdm
    import pandas as pd
    from old_code.gtheory import flexibility_consec_time_windows
    from old_code.gtheory.utils import assign_pair
    save_pickle_name = sort_path(glob(f"{pt['dir_root']}/S01/*/community_theta.pickle"))[0]
    df_event_path = sort_path(glob(f"{pt['dir_root']}/S01/*/filtered-event.feather"))[0]
    df_event = pd.read_feather(df_event_path)
    mevent = df_event['ncluster_3']
    df_ls = pickle.load(open(save_pickle_name, "rb"))
    
    mm = [flexibility_consec_time_windows(np.transpose(df_ls_each)) for df_ls_each in tqdm(df_ls)]
    df = assign_pair(mm, mevent, len(mm[0]))
    df.to_feather('to_plot.feather')
    # df_event['flexibility_consec'] = pd.DataFrame(np.array(mm).reshape(-1,1))
    n = 1


def plot_flexibility():
    '''
    I have not run or validate this function as of 02 December 2021

    '''
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    # {0:'m',1:'gray',2:'r'}
    legend_elements = [Patch(facecolor='m', edgecolor='m', label='Cluster 0', alpha=0.3),
        Patch(facecolor='gray', edgecolor='gray', label='Cluster 1', alpha=0.3),
        Patch(facecolor='r', edgecolor='r', label='Cluster 2', alpha=0.3)]
    
    # Create the figure
    fig, ax = plt.subplots()
    ax.legend(handles=legend_elements, loc='center')
    df = pd.read_feather('to_plot.feather')
    df.reset_index(inplace=True)
    df = df.head(63)
    # df=df.iloc[24:63]
    df2 = df.iloc[::3, :]
    df.rename(columns={'group': 'type'}, inplace=True)
    g = sns.lineplot(data=df, x="index", y="nval", marker='o')
    df2['index'] = df2['index'] + 1.5
    nxtick = df2['index'].values.tolist()
    g.set_xticks(nxtick)  # <--- set the ticks first
    nxlabel = df2['epoch_seq'].values.tolist()
    g.set_xticklabels(nxlabel)
    overlay = {0: 'm', 1: 'gray', 2: 'r'}
    
    for i in np.arange(0, len(df), 3):
        tmp = df.iloc[i:i + 3, :]
        v = overlay.get(tmp.type.unique()[0])
        g.axvspan(min(tmp.index), max(tmp.index) + 1, color=v, alpha=0.3)
        g.text(((min(tmp.index) + max(tmp.index) + 1) / 2) - 1, 0.05,
            'type {}'.format(tmp.type.unique()[0]), fontsize=6)
    
    g.set(xlabel='epoch event', ylabel='Flexibility')
    g.legend(handles=legend_elements, loc='best')
    plt.show()

def filter_epoch_to_remove_mid_condition (debug=True):
    '''
    When first import events from epoch, all event 231,232,233,234 were included in the <raw-event.feather>.
    For rt_local calculation for each deviation onset event,we need the full cycle id(251,252,253,254). Apart from
    rt_local calculation, the id 253 and 254 serve no purpose in the subsequent analysis. Hence, at this stage,
    we need to remove this orphan events and mark the process as 'USER REASON'.
    
    Hence, since we are dropping some events based on certain condition, we rename the file as filtered-event.feather).
    
    '''
    from glob import glob
    from old_code.gtheory import config as cf
    from tqdm import tqdm
    from joblib import Parallel, delayed
    from old_code.gtheory.utils.misc import sort_path
    from old_code.gtheory import drop_epoch_based_condition
    pt = cf.root_path()
    njob = 1
    filter_mid=True
    debug=True
    if debug:
        # opath=['/mnt/d/data_set/sustained_attention_driving/S01/060227n/community_leiden/epoch_cp/coh_theta_epoch_cp.npz']
        logging.info(f'Debug Mood for the CSD at {now.strftime("%d_%m_%Y_%H_%M_%S")}')
        path_epoch = sort_path(glob(f"{pt['dir_root']}/*/*/{pt['prep_epo_fn']}"))[0]
        fpath = sort_path(glob(f"{pt['dir_root']}/*/*/{pt['filtered_event_fn']}"))[0]
        drop_epoch_based_condition(path_epoch, fpath, pt,filter_mid)
    else:
        path_epoch = sort_path(glob(f"{pt['dir_root']}/*/*/{pt['prep_epo_fn']}"))
        
        fpath = sort_path(glob(f"{pt['dir_root']}/*/*/{pt['filtered_event_fn']}"))
        if njob != 1:
            logging.info(
                f'Process: Remove overlap events for a total  {len(fpath)} subjects in parallel mode')
            Parallel(n_jobs=-1, verbose=50)(
                delayed(drop_epoch_based_condition)(path_epoch_, fpath_, pt,filter_mid) for path_epoch_, fpath_
                    in zip(path_epoch,
                    fpath))
        else:
            logging.info(
                f'Process: Remove overlap events for a total  {len(fpath)} subjects in single cpu mode')
            [drop_epoch_based_condition(path_epoch_, fpath_, pt,filter_mid) for path_epoch_, fpath_ in
                tqdm(zip(path_epoch, fpath), desc='Filter only event')]

def epoch_to_epochs(**kwargs):
    # 'theta','alpha','beta','gamma'
    for fband_max_windows in ['theta','alpha','beta','gamma']:
        _epoch_to_epochs (fband_max_windows,**kwargs)

def make_connectivity(**kwargs):
    # 'theta','alpha','beta','gamma'
    for fband_max_windows in ['theta','alpha','beta','gamma']:
        _make_connectivity(fband_max_windows,**kwargs)
def get_extreme_event(df):
    nclus='ncluster_5'
    ls_val=[0,4]
    df_s = df [df[nclus].isin(ls_val)].reset_index(drop=True)
    
def get_extreme_excel_report(**kwargs):
    from glob import glob
    from old_code.gtheory.utils.misc import sort_path
    from old_code.gtheory import config as cf
    import pandas as pd
    import numpy as np

    pt = cf.root_path (**kwargs)
    
    psbjs = sort_path ( glob ( f"{pt ['dir_root']}/*/*/{pt['kmean_c_fn']}" ) )


    # psbjs = list(filter(None.__ne__, [filter_path(fpath,**kwargs) for fpath in psbjs_all]))
    session = kwargs.get('session', None)
    
    # if  session is None:
    spath = f'{pt ["dir_root"]}/dataframes.xlsx'
    nclus='ncluster_5'
    # else:
    #     spath = f'{pt ["dir_root"]}/dataframes_ss_{session}.xlsx'
    #     nclus=f'ncluster_5_ss_{session}'

    
    
    
    df_ls=[]

    
    ls_val=[0,4]
    for fpath in psbjs:
        sbj = findall ( r's\d{2}', fpath, flags=re.IGNORECASE ) [0].upper ()
        
        
        
        sub_session = findall ( r'\d{6}', fpath ) [0]
        sess_N=fpath.split(sub_session)[-1].split(fpath)[0].split(os.sep)[0]
        dcomb=sub_session+sess_N
        df = pd.read_pickle ( fpath )
    
        df ['sbj_id'] = sbj
        df ['sub_session'] = sess_N
        df ['sess_N'] = sub_session
        df['dcomb']=dcomb

        df_s = df [df[nclus].isin(ls_val)].reset_index(drop=True)
        df_ls.append(df_s)
        j=1
    df = pd.concat ( df_ls ).reset_index ( drop=True )


    df = df [['sbj_id', nclus, 'type_exp', 'sub_session','sess_N','dcomb']]
    df ['nval'] = 1
    df_opt = pd.pivot_table(df, values='nval', index=['sbj_id', 'sub_session','sess_N','type_exp','dcomb'],
                    columns=[nclus], aggfunc=np.sum,margins=True)
    
    df_opt['ratio'] = (df_opt[0]/df_opt[4]).where(cond=df_opt[0]-df_opt[4]>=0, other=df_opt[4]/df_opt[0])
    # df_opt['test']=df_opt[0]/df_opt[4]
    # df_opt['test']=[df_opt[0]/df_opt[4] if df_opt[0]-df_opt[4] > 0 else 'green' for x in df['Set']]
    #
    pd.set_option('display.max_columns', None)  # or 1000
    pd.set_option('display.max_rows', None)  # or 1000
    pd.set_option('display.max_colwidth', None)  # or 199
    writer = pd.ExcelWriter(spath, engine='xlsxwriter')
    df_opt.to_excel(writer,sheet_name='Summary report')

    writer.save()
    
def get_excel_report(debug=False,fname=None):
    from glob import glob
    from old_code.gtheory.utils.misc import sort_path
    from old_code.gtheory.reporting import get_cluster_per_subject
    from old_code.gtheory import config as cf
    import pandas as pd
    pd.set_option('display.max_columns', None)  # or 1000
    pd.set_option('display.max_rows', None)  # or 1000
    pd.set_option('display.max_colwidth', None)  # or 199
    pt = cf.root_path ()
    all_df=[]
    if fname is None:
        fevent='filtered_mid-event'
    else:
        fevent=fname
    xpath = sort_path ( glob ( f"{pt ['dir_root']}/*/*/{fevent}.pkl" ) )
    writer = pd.ExcelWriter(f'{pt ["dir_root"]}/dataframes.xlsx', engine='xlsxwriter')
    for nclus in [2,3,4,5]:
        ncluster = f'ncluster_{nclus}'
        print(f'Number cluster: {ncluster}')
        
        df=get_cluster_per_subject ( xpath, ncluster, psave=None )
        df.to_excel(writer,sheet_name=ncluster)
        # print(tabulate(df, headers='keys', tablefmt='psql'))
    writer.save()
# extract_delete_zip ()
# 'n_case_A'
load_to_epoch (debug=False,session='n_case_A') # Produced ('raw-event.feather','prep-epo.fif')
cal_reaction_time (debug=False,session='n_case_A') # Modified the file ('raw-event.feather')
get_reaction_time_global (debug=False,session='n_case_A') # Modified the file ('raw-event.feather')


#### You should consider about this
remove_overlap_events(session='n_case_A') # Produced ('filtered-event.feather')
#
get_cluster_model (session='n_case_A')
cluster_assignment (session='n_case_A') # This can cause unbalance event number between epochs and df events
get_extreme_excel_report(debug=False,session='n_case_A') # Get info how many being remove
get_cluster_per_subject_report (session='n_case_A')


# filter_epoch_to_fulfiled_condition (debug=debug_status,session='n_case_A',filter_mid=True) # Produced ('visualise_drop_index_inred',
# 'filtered-epo.fif')



