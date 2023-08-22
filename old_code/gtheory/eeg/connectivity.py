# from mne_connectivity import spectral_connectivity
from os.path import (join,split)
from pickle import (HIGHEST_PROTOCOL,dump,load)

import numpy as np
# from mne.connectivity import spectral_connectivity
from tqdm import tqdm

from old_code.gtheory import config as cf
from old_code.gtheory.utils.dataframe import load_events_label
from old_code.gtheory.utils.misc import check_make_folder

'''
1) https://github.com/mne-tools/mne-connectivity
2) https://github.com/balandongiv/spectral_connectivity
3) https://github.com/scot-dev/scot
4) https://github.com/neuropycon
'''


def define_freq_bands (set_data):
    # Define freq bands
    n=tuple (
        [list ( set_data.values () ) [f] [0] for f in
         range ( len ( set_data ) )] ), tuple (
        [list ( set_data.values () ) [f] [1] for f in range ( len ( set_data  ) )] )
    return n


def get_spectral_connectivity (path_epoch,fband_dict,fbandname,pt,bsave=False,**kwargs):
    '''

    Parameters
    ----------
    path_epoch
    bsave

    Returns
    -------
    (nepoch, nslice, nnodes, nnodes, nfbands) Epoch, windows, Nchannel,Nchannel, fbands
    '''
    
    label_epoch=load_events_label(path_epoch,pt,clustertype=None,spliter_s='overlapped_epoch',**kwargs)
    fmin, fmax = define_freq_bands (fband_dict)
    params = dict ( connectivity_methods=cf.connectivity_methods, connectivity_mode='multitaper', fmin=fmin,
        fmax=fmax,path_info=pt )
    
    # filter_mid= kwargs.get('filter_mid',None)
    # save_data(path_epoch,np.array([1]),['r'],params,fbandname,filter_mid)
    
    ls_epoch = load(open(path_epoch, "rb"))
    
    # ep_len=len(ls_epoch)
    # T=label_epoch.shape[0]
    if len(ls_epoch) !=label_epoch.shape[0]:
        #Sanity Check
        raise ValueError (f'Number of epochs and label is diffrent: {path_epoch}')
    
    
    # Iterate each epoch
    con =[_connectivity_per_epochs ( epochs, params ) for epochs in tqdm(ls_epoch)]
    con=np.array(con)
    k=con.shape[0]
    if k !=label_epoch.shape[0]:
        #Sanity Check
        print(path_epoch)
        raise ('Number of epochs and label is diffrent')
    if bsave:
        save_data(path_epoch,con,ls_epoch[0].ch_names,params)

    return con


def save_data(fpath,con_arr,ch_name,param):
    '''
    https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html
    Returns
    -------
    '''
    save_as_pickle=False # Force to save as npz
    if save_as_pickle:
        path_pickle = fpath.replace("epo", "con")
        with open(path_pickle, 'wb') as handle:
            dump([con_arr,ch_name,param], handle, protocol=HIGHEST_PROTOCOL)
    else:
        root_folder,fname = split ( fpath)
        root_folder,_=split(root_folder)
        # Assuming we going to experiment diffrent type connectivity thresholding, better create dedicated
        # connectivity folder
        fdir=join(root_folder,'connectivity')
        check_make_folder(fdir)
        # if not filter_mid:
        spath=join(fdir,param['path_info']['connectivity_result_fn'])
        # else:
        #     spath=join(fdir,f"{fbandname}_mid_{param['path_info']['connectivity_result_fn']}")
        np.savez_compressed(spath, con_arr=con_arr, ch_name=ch_name,param=param)
# f'{fband}_mid_connectivity_result.npz'

def _connectivity_per_epochs (epochs, params):
    # # Calculate PLV for each trial
    # all_con=[cal_connectivity (epoch, params,epochs.info ['sfreq']) for epoch in epochs]
    return [cal_connectivity (epoch, params,epochs.info ['sfreq']) for epoch in epochs]


def cal_connectivity (epoch, params,sfreq):
    '''
    https://github.com/mne-tools/mne-connectivity/blob/main/examples/mne_inverse_coherence_epochs.py
    :param epoch:
    :param params:
    :return:
    '''
    
    # Need to install new package. latest mne version removed this
    con, freqs, times, n_epochs, n_tapers = spectral_connectivity (
            [epoch], method=params ['connectivity_methods'], mode=params ['connectivity_mode'],
            sfreq=sfreq, fmin=params ['fmin'], fmax=params ['fmax'],
            faverage=True, verbose=0 )
    return con

