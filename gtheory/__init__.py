# from ._version import __version__

from .cluster import (fit_kmeans,predict_kmeans_model,viz_prediction_kmeans)
# . then folder name followed by the python file name, and import the specific function

from .download import (extract_remove_zip,_get_file_web)
from .eeg import (load_raw_to_epochs,sliding_epoch_to_epochs,make_csd,get_spectral_connectivity,
  drop_epoch_based_condition)
# from .grapht import (allegiance,flexibility_index,community_membership,gt_leidenalg,
#                      flexibility_consec_time_windows,get_community_each_epoch,
#                      supra_adj_matrix_list,comm_measure,comm_measure_leiden,consensus_over_epoch_nslice,
#                      modular_allegiance,ncommunities_slices,combine_ncommunities_sbj_epoch_identity,
#                      flexibilities_arr,get_sub_zrand,
#                      iterate_over_each_label_consensus_partition,
#                      comm_measure_louvain)
# from .grapht.dummy_file import dum_time_slice
# from .grapht.ncat import *
# from .mock_data import (md_modularity_fragmentation,md_flexible_clubs,md_parasweep_leiden)
from .reporting import (get_cluster_per_subject,make_generic_report_layout)
from .task_state import (reaction_time,get_global,rejected_events)
from .utils.dataframe import (assign_pair,save_list_df_multiple_sheets,load_melt_df_barplot,load_events_label)
from .utils.misc import load_json,get_omega_gamma,filter_path
from .utils.misc import (sort_path,split_id_session,extract_motion_normal,node_subsystem,drop_overlap_event)
from .utils.nparray import extract_maximum_size_sublist
from .utils.nparray import (slice_array_with_list,between_list_diff_index)
from .viz import (viz_isometric_heatmap,viz_connectivity_circle,viz_comm_measure_club,
  scatterplot_correlation_comm_measure,scatterplot_correlation_ncommunities)
