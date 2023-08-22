import platform
from os.path import join

ncluster = ['ncluster_2', 'ncluster_3', 'ncluster_4', 'ncluster_5']
# freq_bands = {"delta": [1.25, 4.0], "theta": [4.0, 8.0], "alpha": [8.0, 13.0],
#               "beta": [13.0, 30.0], "gamma": [30.0, 49.0]}
freq_bands = {"theta": [4.0, 8.0], "alpha": [8.0, 13.0],
              "beta": [13.0, 30.0], "gamma": [30.0, 49.0]}

fband_con_max=dict(theta= {'fband':['theta','alpha','beta','gamma'],'window_epoch_size':2,'freq_bands' : {"theta": [4.0, 8.0], "alpha": [8.0, 13.0],"beta": [13.0, 30.0], "gamma": [30.0, 49.0]}},
                   alpha= {'fband':['alpha','beta','gamma'],'window_epoch_size':0.8,'freq_bands' : {"alpha": [8.0, 13.0],"beta": [13.0, 30.0], "gamma": [30.0, 49.0]}},
                   beta= {'fband':['beta','gamma'],'window_epoch_size':0.4,'freq_bands' : {"beta": [13.0, 30.0], "gamma": [30.0, 49.0]}},
                   gamma= {'fband':['gamma'],'window_epoch_size':0.2,'freq_bands' : {"gamma": [30.0, 49.0]}},
                   test_band= {'fband':['gamma'],'window_epoch_size':2,'freq_bands' : {"gamma": [30.0, 49.0]}})

# fband_con_max=dict(theta= {'fband':['theta','alpha'],'window_epoch_size':2,'freq_bands' : {"theta": [4.0, 8.0], "alpha": [8.0, 13.0],"beta": [13.0, 30.0], "gamma": [30.0, 49.0]}},
#   alpha= {'fband':['alpha','beta'],'window_epoch_size':0.8,'freq_bands' : {"alpha": [8.0, 13.0],"beta": [13.0, 30.0]}},
#   beta= {'fband':['beta','gamma'],'window_epoch_size':0.4,'freq_bands' : {"beta": [13.0, 30.0], "gamma": [30.0, 49.0]}},
#   gamma= {'fband':['gamma'],'window_epoch_size':0.2,'freq_bands' : {"gamma": [30.0, 49.0]}})
## alpha 1/8= 0.125s x 6cycle =0.75s >>4sec/0.8 ==5 windows
## beta 1/13=0.077s x 6cycle =0.4615  >>4 sec/0.4 ==10 windows
## gamma 1/30=0.033 x 6cycle = 0.2 >>4 sec /0.2 ==20 windows
"""
Case consideration:
window_epoch_size=10
"""
# window_epoch_size = 2  >> please change this var under the function below

# Mainly to consider: pli,wpli. and there is special justification why WPLI was selected
'''
1) Reproducibility of functional connectivity and graph measures based on the phase lag index (PLI) and weighted phase
lag index (wPLI) derived from high resolution EEG

2) Alternating Dynamics of Segregation and Integration in Human EEG Functional Networks During Working-memory Task

# See: https://mne.discourse.group/t/range-of-diffrent-connectivity-measure/3579?u=balandongiv
pli2_unbiased and the wpli2_debiased produced by the mne.connectivity.spectral_connectivity [-1,1].

produced by the mne.connectivity.spectral_connectivity are within the range [0,1].
['coh','cohy' ,'imcoh','plv','ciplv','ppc','pli','wpli']

Ignore
'cohy' since it make the other con as an imgiry type
'ppc','pli','pli2_unbiased', remove since there is NaN Weight

already:
'''
connectivity_methods = ['coh', 'imcoh', 'plv', 'ciplv', 'wpli', 'wpli2_debiased']
# connectivity_methods = ['coh', 'imcoh']
para_setting = dict(nsubject='all',
                    # fband_list=['theta'],
                    # connectivity_methods=['coh','plv'],
                    fband_list=list(freq_bands.keys()),  # List of fbands as define in config.py
                    connectivity_methods=connectivity_methods,
                    # List of con method as define in config.py
                    slice_limits=10,
                    # slice_limits=15 # Speed consideration, leaden can operate at best about 150.
                    # slices. Used increment of 30 for each 1minute
                    fname_para_sweeping='para_sweeping',  # Dir name to store each subanalysis
                    nrep=100, #100
                    njob=1,
                    romega=(-1, 1, 0.02),  #romega=(0.0, 3, 0.1),
                    rgamma=(-1, 1, 0.01), #rgamma=(0.0, 3, 0.1),
                    # romega=(0.0,2,0.4),
                    # rgamma=(0.0,2,0.4),
                    # romega=(0.01,1,0.5),
                    # rgamma=(0.01,1,0.5),
                    omega=0.41,
                    gamma=0.41,
                    gamma_cp=1,
                    flush_batch_files=True, local_save=True, flush_previous_finding=True)

# bin_type=None, tbinarise=None, tcomm=None,filter_mid=None,fband=None,
def root_path(**kwargs):

    # Get the default value here
    tcomm = kwargs.get('tcomm',None)
    bin_type = kwargs.get('bin_type',None)
    tbinarise = kwargs.get('tbinarise',None)
    filter_mid= kwargs.get('filter_mid',None)
    fband= kwargs.get('fband',None)
    session= kwargs.get('session',None)
    # myhost = os.uname () [1]



    overlap_window = 0  # size of overlap when sliding the windows. Use for create epoch2epochs
    # window_epoch_size = 2  # new epoch size. Remember we slide the original epoch


    myhost = platform.uname()



    if myhost.node == 'rpb' and myhost[0] == 'Windows':

        dir_root = r'C:\Users\balan\OneDrive\Desktop\data_sustained'
    elif myhost.node == 'DESKTOP-H8CEMP8' and myhost[0] == 'Linux':

        dir_root = '/mnt/e/sustained_attention_driving - Copy/'
    elif myhost.node == 'DESKTOP-H8CEMP8' and myhost[0] == 'Windows':
        dir_root = 'E:\sustained_attention_driving - Copy'
    else:
        dir_root = 'REPLACE YOUR PATH HERE'



    if tcomm is None:
        ftcomm = 'NA'
    elif tcomm == 'leiden':
        ftcomm = 'leiden'
    elif tcomm == 'louvain':
        ftcomm = 'louvain'

    if (bin_type is not None) and (tbinarise is not None) and (fband is not None) :
        connectivity_result_fn = 'connectivity_result.npz'
        f_comm = f'{ftcomm}'
    elif filter_mid is True and session is not None:
        connectivity_result_fn = f'{fband}_{session}_mid_connectivity_result.npz'
        f_comm = f'{ftcomm}_{session}_{fband}'
    else:
        connectivity_result_fn = f'connectivity_result_{bin_type}_{tbinarise}.npz'
        f_comm = f'{ftcomm}_{bin_type}_{tbinarise}'


    if session is None:
        event_log='raw-event_def.pkl'
        kmean_c_fn = 'kmean_c-event_def.pkl'
        cluster_kmeans_distribution_fn = 'all_subject_cluster_distribution.xlsx'
        filtered_epo_fn='filtered_04-epo.fif' if filter_mid else 'filtered-epo.fif'
        sel_events='sel_events_04.pkl' if filter_mid else 'sel_events_all.pkl'
        csd_epo_fn='csd_04-epo.fif' if filter_mid else 'csd-epo.fif'
    else:
        event_log=f'raw-event_{session}.pkl'
        kmean_c_fn = f'kmean_c-event_{session}.pkl'
        cluster_kmeans_distribution_fn = f'all_subject_cluster_distribution_{session}.xlsx'
        filtered_epo_fn=f'filtered_04_{session}-epo.fif' if filter_mid else f'filtered_{session}-epo.fif'
        sel_events=f'sel_events_04_{session}.pkl' if filter_mid else f'sel_events_all_{session}.pkl'

        csd_epo_fn =f'csd_04_{session}-epo.fif' if filter_mid else f'csd_{session}-epo.fif'





    if fband is not None and session is not None:
        overlapped_epo_fn=f"overlapped_04_{fband}_{session}-epo.pickle" if filter_mid else f"overlapped_{fband}_{session}-epo.pickle"

    elif fband is not None:
        raise Warning ('My first time seing you')

    else:
        overlapped_epo_fn=''




    connectivity_result = join(dir_root, '*', '*', '*', connectivity_result_fn)


    event_log_lgrt='event_log_lgrt.pkl'
    ### To investigate how to change this
    community_leiden = 'community_leiden'  # Example > '/mnt/d/data_set/sustained_attention_driving/S01/051017m/community_leiden/8a_cp_over_epoch_condi/coh_delta_cp_over_epoch_condi.npz'

    tmin_do = -4  # how long before deviation onse. t
    event_id = {'251': 1,
                '252': 2}  # Event ID to extract from epoch event. 1:left deviaiton, 2:right deviation
    fsampling = 256
    '''
  History overlap_threshold
  Tried with 16s, as expected, this will remove most of the data.
  '''
    overlap_threshold = 4  # Unit is in second. Remove event if the preceeding events is within this overlap_threshold

    kmean_model = join(dir_root, 'kmeans_model', '')


    source_local = join(dir_root, 'source_local', '')


    report = join(dir_root, 'report', '')

    stat_univariate = join(dir_root, 'stat_univariate')





    con_bct = join(dir_root, 'con_bct', '')


    community_measure = join(dir_root, 'comm_analysis', f_comm, '')

    para_betzel=join(dir_root, 'comm_analysis', 'betzel_sweeping',f_comm, )
    para_sweep = join(community_measure, 'para_sweep', '')


    lamda_omega_space = join(community_measure, 'lamda_omega_space', '')

    #### community_measure_leiden

    cp_slice = join(community_measure, '6a', 'step2_cp_slice', '') # Step 2 of 6a


    cp_window = join(community_measure, '6a', 'step3_cp_window', '') # Step 3 of 6a


    cp_conditions = join(community_measure, '6a', 'step4_cp_conditions') # Step 4  of 6a


    epoch_cp_store = join(community_measure, '6a', 'epoch_cp', '')

    #####

    all_epoch_cp = join(community_measure, '6b', 'step1_all_epoch_cp', '') # Data from step 1 of 6b


    condi_epoch_cp = join(community_measure, '6b', 'step2_condi_epoch_cp', '') # Data from step 2 of 6b

    ####

    cognitive_engagement = join(community_measure, 'cognitive_engagement', '') # Folder 8a and use to store data from step 1

    cp_over_epoch_condi = '8a_cp_over_epoch_condi'  # This is here as a way to create subfolder under community_leiden under

    # each sbj folder


    module_reconfig_flexibility = join(community_measure, 'module_reconfig_flexibility', '')


    module_integration = join(community_measure, 'module_integration', '')


    club_flexible = join(community_measure, 'club_flexible', '')


    club_integration = join(community_measure, 'club_integration', '')


    modularity_fragmentation = join(community_measure, 'modularity_fragmentation', '')

    # subfolder for communities, integration, and flexibility
    # scatter_correlation=f'{community_measure}scatter_correlation/'
    scatter_correlation = join(community_measure, 'scatter_correlatio', '')

    para_sweep_cp = '_para_sweep_consensus_partition.npz'
    para_sweep_comm_measure = '_para_sweep.pkl'
    epoch_cp_fn = '_epoch_cp.npz'  # fname for all each epoch consensus
    cp_slice_fn = '_step2_slice_consensus.pkl'
    cp_window_fn = '_step3_slice_consensus.pkl'
    cp_conditions_fn = '_step4_slice_consensus.pkl'
    raw_event_fn = 'raw-event.pkl'
    cp_epoch_label_fn = '_step1_epoch_label_consensus.pkl'  # used in 6b analysis
    cp_epoch_condition_fn = '_step2_epoch_label_consensus.pkl'  # used in 6b analysis
    cp_over_epoch_condi_fn = '_cp_over_epoch_condi.pkl'
    cognitive_system_engagement_fn = '_step2_cognitive_system_engagement.pkl'
    mod_reconfig_flexi_fn = '_module_reconfig_flexibility.pkl'
    module_integration_fn = '_module_integration.pkl'
    flexible_club_fn = '_flexible_club.pkl'
    integration_club_fn = '_flexible_club.pkl'

    filtered_event_fn = 'filtered-event.pkl'
    filtered_mid_event_fn = 'filtered_mid-event.pkl' # as per Dr Tang suggestion

    # event_cluster_rt_fn='event_cluster_rt.pkl' # We have to create separate file due to events no become out sync

    modularity_fragmentation_fn = '_modularity_fragmentation.pkl'
    para_sweep_cpartition_rep = 'param_sweeping_cpartition_rep.html'
    para_sweep_comm_measure_rep = 'para_sweep_comm_measure_rep.html'
    flexible_club_rep = 'flexible_club_rep.html'
    integration_club_rep = 'integration_club_rep.html'
    modularity_fragmentation_rep = 'modularity_fragmentation_rep.html'
    flexibility_correlation_rep = 'flexibility_correlation_rep.html'
    integration_correlation_rep = 'integration_correlation_rep.html'
    #######
    prep_epo_fn = 'prep-epo.fif'





    ttest_roi_nonroi_cognitive_system_engagement = 'ttest_roi_nonroi_cognitive_system_engagement.xlsx'

    # Reason why we cannot concat all the epochs is because there is duplicate identity accross newly create epoch.
    # Hence we use pickle instead




    # epoch_cp='_epoch_cp.npz'
    epochs_concat_for_csd_evaluation = 'epochs_concat_for_csd_evaluation-epo.fif'

    para_sweep_JS = 'para_sweep.json'
    subject_similarity_JS = 'subject_similarity.json'
    para_sweep_comm_JS = 'para_sweep_comm.json'

    ### PSD

    unifeatuare = 'unifeatuare'
    unifeatuare_events_fn = 'unifeatuare_events.pkl'
    return dict(tmin_do=tmin_do, event_id=event_id, fsampling=fsampling,
                overlap_threshold=overlap_threshold,
                dir_root=dir_root,session=session,
                stat_univariate=stat_univariate,
                con_bct=con_bct, community_leiden=community_leiden,
                source_local=source_local,event_log=event_log,
                kmean_model=kmean_model, f_comm=f_comm,
                report=report, connectivity_result_fn=connectivity_result_fn,
                community_measure=community_measure,
                para_sweep=para_sweep,filtered_mid_event_fn=filtered_mid_event_fn,
                lamda_omega_space=lamda_omega_space,
                cp_window=cp_window, ftcomm=ftcomm,sel_events=sel_events,
                condi_epoch_cp=condi_epoch_cp,event_log_lgrt=event_log_lgrt,
                cognitive_engagement=cognitive_engagement,
                cp_over_epoch_condi=cp_over_epoch_condi,
                module_integration=module_integration,
                epochs_concat_for_csd_evaluation=epochs_concat_for_csd_evaluation,
                club_flexible=club_flexible, prep_epo_fn=prep_epo_fn,
                epoch_cp_fn=epoch_cp_fn, filtered_epo_fn=filtered_epo_fn, csd_epo_fn=csd_epo_fn,
                overlapped_epo_fn=overlapped_epo_fn,
                cp_epoch_label_fn=cp_epoch_label_fn,
                cp_epoch_condition_fn=cp_epoch_condition_fn, filtered_event_fn=filtered_event_fn,
                epoch_cp_store=epoch_cp_store, club_integration=club_integration,
                integration_club_fn=integration_club_fn,
                para_betzel=para_betzel,
                cp_slice=cp_slice,overlap_window=overlap_window,
                ttest_roi_nonroi_cognitive_system_engagement=ttest_roi_nonroi_cognitive_system_engagement,
                cluster_kmeans_distribution_fn=cluster_kmeans_distribution_fn,
                modularity_fragmentation_fn=modularity_fragmentation_fn,
                flexible_club_fn=flexible_club_fn, subject_similarity_JS=subject_similarity_JS,
                para_sweep_comm_JS=para_sweep_comm_JS,
                cp_conditions=cp_conditions,
                modularity_fragmentation=modularity_fragmentation,
                scatter_correlation=scatter_correlation,
                connectivity_result=connectivity_result,
                all_epoch_cp=all_epoch_cp,
                cp_slice_fn=cp_slice_fn,
                cp_window_fn=cp_window_fn,
                cp_conditions_fn=cp_conditions_fn,
                cp_over_epoch_condi_fn=cp_over_epoch_condi_fn,
                cognitive_system_engagement_fn=cognitive_system_engagement_fn,
                mod_reconfig_flexi_fn=mod_reconfig_flexi_fn,
                module_reconfig_flexibility=module_reconfig_flexibility,
                module_integration_fn=module_integration_fn,
                raw_event_fn=raw_event_fn,kmean_c_fn=kmean_c_fn,
                para_sweep_JS=para_sweep_JS,
                unifeatuare=unifeatuare, unifeatuare_events_fn=unifeatuare_events_fn,
                comm_measure=dict(para_sweep_cp=para_sweep_cp,
                                  para_sweep_comm_measure=para_sweep_comm_measure),
                rep=dict(para_sweep_cpartition=para_sweep_cpartition_rep,
                         modularity_fragmentation_rep=modularity_fragmentation_rep,
                         para_sweep_comm_measure=para_sweep_comm_measure_rep,
                         flexible_club_rep=flexible_club_rep,
                         integration_club_rep=integration_club_rep,
                         flexibility_correlation_rep=flexibility_correlation_rep,
                         integration_correlation_rep=integration_correlation_rep))
