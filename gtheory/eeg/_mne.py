import pickle
from os.path import (join,split)

import mne
import numpy as np
import pandas as pd
from mne import read_epochs

from gtheory.task_state import rejected_events
from gtheory.utils.dataframe import load_events_label
from gtheory.utils.misc import check_make_folder
from gtheory.utils.misc import (extract_motion_normal)
import os
'''
https://www.youtube.com/watch?v=CodQ5-pmXdQ
'''


def get_event(raw, df_path, sfreq, type_expt, save_event=True):
    '''

    Of note, the next trial occurred within a 5–10 second interval after finishing
    the current trial, during which the subject had to maneuverer the car back to the
    centre line of the third car lane. If the participant fell asleep during the
    experiment, no feedback was provided to alert him/her.
    :param raw:
    :param df_path:
    :param sfreq:
    :return:
    '''
    events = mne.events_from_annotations(raw)
    df = pd.DataFrame(events[0], columns=['timepoint', 'duration', 'event_key'])
    df["event_id"] = df["event_key"].map(dict((value, key) for key, value in events[1].items()))
    df['sfreq'] = sfreq
    df['type_exp'] = type_expt
    if save_event:
        df.to_pickle(df_path)
    else:
        return df


def load_raw_to_epochs(fpath, pt, debug=True, path_epochs=None):
    '''
    Load the .set continous signal, assign montage and extract the epochs.

    When epoching, the tmin is set to -8 since the random pertubation occured
    between 10-8 second
    :param path_epochs:
    :param fpath:
    :param debug:
    :return
    '''
    if path_epochs is None:

        root_folder, fname = split(fpath)
        file_path_store = join(root_folder, pt['prep_epo_fn'])
        df_path = join(root_folder, pt['raw_event_fn'])
        expt_session = extract_motion_normal(fpath)


    else:
        file_path_store = path_epochs

    if debug:
        raw = mne.io.read_raw_eeglab(fpath, preload=True).crop(tmax=160)
    else:
        raw = mne.io.read_raw_eeglab(fpath, preload=True)
    # raw.plot(show=True)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, match_case=False)
    raw = raw.resample(pt['fsampling'], npad="auto", n_jobs=1)
    re_referenced_raw, ref_data = mne.set_eeg_reference(raw, ref_channels='average',
                                                        copy=False, projection=True)

    # Space and speed consideration, we filter only specific events (i.e., 251,252)
    events = mne.events_from_annotations(re_referenced_raw, event_id=pt['event_id'])

    epochs = mne.Epochs(re_referenced_raw, events[0], preload=True, tmin=pt['tmin_do'],
                        tmax=0, event_id=pt['event_id'], baseline=None)

    epochs.save(file_path_store, overwrite=True)
    ## While it is desireable to usThe variable `epochs` is a filtered version of <re_referenced_raw>. Specifically,
    # only event register under

    # Extract events for reaction time calculation amd save as <raw-event.feather>
    get_event(re_referenced_raw, df_path, raw.info['sfreq'], expt_session)


def sliding_epoch_to_epochs(path_epoch,pt, overlap_window=1, window_epoch_size=2.5,
                            return_ep=False,**kwargs):
    '''

    I have visually check (01 December 2021), the figure epoch and epochs_slices is tally. More importantly, the first
    epoch_slice is
    window far away from the deviation onset, and the last epoch_slice is the closest to the deviation onset. The
    snipet for the verification available in _mne.py of the function sanity_check_original_epoch_to_slicesepochs()

    # Create epochs from epoch using sliding window approach.
    # overlap_window: size of the overlapping window
    # Cross reference between filtered-event.feather and the overlap epoch is thru ep.info ['description'] .
    # From this, we can identified which cluster the ep belong to
    '''

    root_folder, fname = split(path_epoch)
    

    # file_ext=f"{kwargs['fband']}_{pt['overlapped_epo_fn']}"


    pdf=os.path.join(root_folder,pt['sel_events'])
    label_epoch=pd.read_pickle(pdf)
    
    spath=join(root_folder, 'overlapped_epoch')
    check_make_folder(spath)
    file_path_store = join(spath, pt['overlapped_epo_fn'])
    # label_epoch = load_events_label(path_epoch, pt, clustertype=None, spliter_s=fsplit,filter_mid=filter_mid)
    epoch_set = mne.read_epochs(path_epoch, preload=False)

    ep_ls = []
    for epochs_data, t in zip(epoch_set, epoch_set.events):
        ep = mne.make_fixed_length_epochs(mne.io.RawArray(epochs_data, epoch_set.info),
                                          duration=window_epoch_size, preload=True,
                                          overlap=overlap_window)
        ep.events[:, -1] = np.arange(0, len(ep)) + (251 if t[-1] == 1 else 252 * 10)
        ep.info['description'] = str(t)
        n = ep.events[:, -1].tolist()
        ep.event_id = dict(zip(map(str, n), n))
        ep_ls.append(ep)


    hhh=len(ep_ls)
    hxx=label_epoch.shape[0]
    if len(ep_ls) != label_epoch.shape[0]:
        # Sanity Check
        raise ValueError(f'Number of epochs and label is diffrent: {path_epoch}')

    # Reason why we cannot concat all the epochs is because there is duplicate identity across newly create epoch.
    # Hence we use pickle instead.
    with open(file_path_store, 'wb') as handle:
        pickle.dump(ep_ls, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if return_ep:
        return mne.concatenate_epochs(ep_ls)


def sanity_check_original_epoch_to_slicesepochs():
    '''
    This provide the means to visually validate whether the sliding window proceed as expected or not. From the plot,
    we also can have the idea about the sequence of the new epochs indexing. Such that, first epoch_slice is the
      window that is far away from the deviation onset, and the last epoch_slice is the closest to the deviation onset
    '''
    import mne
    import numpy as np

    path_epoch = '/mnt/d/data_set/sustained_attention_driving - Copy/S01/051017m/csd-epo.fif'
    window_epoch_size = 2
    overlap_window = 0
    epoch_set = mne.read_epochs(path_epoch, preload=False)

    for epochs_data, t in zip(epoch_set, epoch_set.events):
        ep = mne.make_fixed_length_epochs(mne.io.RawArray(epochs_data, epoch_set.info),
                                          duration=window_epoch_size, preload=True,
                                          overlap=overlap_window)
        ep.events[:, -1] = np.arange(0, len(ep)) + (251 if t[-1] == 1 else 252 * 10)
        ep.info['description'] = str(t)
        n = ep.events[:, -1].tolist()
        ep.event_id = dict(zip(map(str, n), n))
        ##################################
        '''
        I have visually check, the figure epoch and epochs_slices is tally. More importantly, the first epoch_slice is
        window far away from the deviation onset, and the last epoch_slice is the closest to the deviation onset
        '''
        raw = mne.io.RawArray(epochs_data, epoch_set.info)
        raw.plot()  # The original epoch
        ep.plot()  # epochs of the single epoch


def drop_epoch_based_condition(path_epoch, fpath, pt,**kwargs):
    '''
    1) When first import events from epoch, all event 231,232,233,234 were included in the <raw-event.feather>.
    For rt_local calculation for each deviation onset event,we need the full cycle id (251,252,253,254). Apart from
    rt_local calculation, the id 253 and 254 serve no purpose in the subsequent analysis. Hence, at this stage,
    we need to remove this orphan events and mark the process as 'USER REASON'.

    2) Apart from that, it is also desireable to remove overlap events. See: drop_overlap_event()

    COnsidering (1) and (2),such that we are dropping some events based on certain condition, we rename the file as
    filtered-event.feather).

    We also make a epoch file which we name as <filtered_epo>

    '''

    root_folder, fname_ext = split(path_epoch)
    
    # if not filter_mid:
    file_path_store = join(root_folder, pt['filtered_epo_fn'])
    path_event_df=join(root_folder, pt['sel_events'])


    
    
    df_event = pd.read_pickle(fpath)
    epochs = read_epochs(path_epoch, preload=False)

    # Derive the orphan events within the epochs. Specifically, cross check with timing in the
    # <filtered-event.feather> and drop epochs_events that did not exist in the <filtered-event.feather>
    kwargs['rfolder']=root_folder
    kwargs['viz']=True
    kwargs['path_event_df']=path_event_df
    
    epochs_drop_idx = rejected_events(epochs.events, df_event,**kwargs)
    # epochs.drop ( epochs_drop_idx, reason='USER REASON' )
    epochs.drop(epochs_drop_idx, reason='BAD')
    label_epoch = load_events_label(path_epoch, pt, clustertype=None, spliter_s='prep-epo.fif',**kwargs)
    ep_len = len(epochs)
    ep_len = len(epochs.selection.tolist())
    df_len = label_epoch.shape[0]
    if len(epochs) != label_epoch.shape[0]:
        # Sanity Check
        raise ValueError(f'Number of epochs and label is diffrent: {path_epoch}')

    epochs.save(file_path_store, overwrite=True)


