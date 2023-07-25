import numpy as np
from gtheory.grapht import find_consensus


def consensus_over_epoch_nslice(S, algo_type,gamma):
    """

    For each subject and each condition, lump all the time slices for all epochs
    Input S: (Node, slice_of_all_epochs)

    Return: Consensus partition of the time slices for all epochs
    """
    # Option A

    consensus_part= find_consensus(S, algo_type,null_func=np.mean,return_agreement=False,seed=None, gamma=gamma)
    return consensus_part

def iterate_over_each_label_consensus_partition(label_epoch, nlabel, epoch_com_arr, algo_type,gamma=1,
 combine_epoch=False):
    '''
    Previously known as iterate_over_each_label_consensus_partition_leidean
    :param label_epoch:
    :param nlabel:
    :param epoch_com_arr:
    :param gamma:
    :return: list of Consensus partition. The list have len equaivalent to number of slices (times or windows).
    Within each list, reside the 1D array with len equivalent to number of nodes. For simplicity, we convert the list into the form
    of numpy array.
    '''
    # Get epoch index that corresponding to the mental condition of interest
    idx_selc = np.where(label_epoch == nlabel)[0]
    idx_for_specific_condition_label = idx_selc.tolist()
    
    # Slice epoch that correspond to specific label
    time_node_arr = epoch_com_arr[idx_for_specific_condition_label] # (epochs, nslice, nnodes)
    
    if not combine_epoch:
        #>> From all epochs, get that a slice at the location idx from each of the epochs, this is produced--(
        # nepochs,nnodes)
        
        # We need to Transpose the time_node_arr[:, nslice_idx, :] since `find_consensus` received input of shape (Nodes,
        # solution/times/nepochs) array_like
        
        cpartition_ls = [find_consensus(time_node_arr[:, nslice_idx, :].T, algo_type,gamma=gamma) for nslice_idx in
            range(time_node_arr.shape[1])]
        return np.array(cpartition_ls)
    else:
        # cpartition_ls = find_consensus(np.concatenate((time_node_arr)).T, gamma=gamma)
        return find_consensus(np.concatenate((time_node_arr)).T, algo_type,gamma=gamma)
        
    

#
# def iterate_over_each_label_consensus_partition(label_epoch, nlabel, epoch_com_arr, Time, Node, gamma=1):
#     idx_selc = np.where(label_epoch == nlabel)[0]
#     b = idx_selc.tolist()
#     expected_output = epoch_com_arr[b, :]
#
#     nshap = expected_output.shape[0] if len(expected_output.shape) >= 2 else 1
#
#     time_node_arr = np.reshape(expected_output, [nshap, Time, Node])
#     # # Sanity check for reshape above. Please do not delete
#     # # all_m_arr=[np.reshape(s, [Time, Node]) for rep,s in enumerate(expected_output)]
#     # # result_sanity_check=np.array_equal(test_a, all_m_arr) # Return true if equal
#
#     # # From all epochs, get that particular slice from each of the epochs
#     cpartition_ls = [find_consensus(time_node_arr[:, nslice_idx, :], gamma=gamma) for nslice_idx in range(Time)]
#     # cpartition_ls = np.array(cpartition_ls)
#     return np.array(cpartition_ls)
