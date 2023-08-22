import os
from os.path import (join,split)

from mne import read_epochs
from mne.preprocessing import compute_current_source_density

import pandas as pd
def _csd (epoch, lambda2=1e-5, stiffness=4):
    # epoch_csd = compute_current_source_density ( epoch, lambda2=lambda2, stiffness=stiffness )
    return compute_current_source_density ( epoch, lambda2=lambda2, stiffness=stiffness )


def make_csd (path_epoch, pt,lambda2=1e-5, stiffness=4, return_ep=False, **kwargs):
    # Used default value  lambda2=1e-5, stiffness=4, n_legendre_terms=50
    

    root_folder,_= split ( path_epoch )
    

    
    pdf=os.path.join(root_folder,pt['sel_events'])
    label_epoch=pd.read_pickle(pdf)
    # label_epoch=load_events_label(path_epoch,pt,clustertype=None,spliter_s='filtered-epo.fif',**kwargs)
    file_path_store = join ( root_folder, pt['csd_epo_fn'] )

      
      
    ep=read_epochs ( path_epoch, preload=False )
    df_len=label_epoch.shape[0]
    ep_len=len(ep.selection.tolist ())

    if ep_len !=label_epoch.shape[0]:
        #Sanity Check
        raise ValueError (f'Number of epochs and label is diffrent: {path_epoch}')
    
    epoch_csd = _csd ( read_epochs ( path_epoch, preload=True ), lambda2=lambda2, stiffness=stiffness )
    epoch_csd.save ( file_path_store, overwrite=True )
    if return_ep:
        return epoch_csd
