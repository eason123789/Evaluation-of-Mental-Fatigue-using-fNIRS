import numpy as np

def extract_maximum_size_sublist(alist_arr):
    '''
    
    Example
    
    np.random.seed(1)
    nnodes,nslice,nepoch,nsubject=30,4,10,4
    # Make sure to use seed(1) to ensure there is incomplete label>> nnodes,nslice,nepoch,nsubject=30,4,10,4
    label_unique=[np.unique(label_epoch).tolist() for label_epoch in
      [np.random.randint(3,size=nepoch) for _ in range(nsubject)]]
      
    consensus_partition_slice_condition_sbj=[[np.random.randint(10,size=(nslice,nnodes)) for _ in range(
        len(nunique_cond))] for _,nunique_cond in zip(range(nsubject),label_unique)]
    
    # filtered list with maximum shape sublist
    arr=get_maximum_size_sublist(consensus_partition_slice_condition_sbj)
    '''
    ## For the time being, we remove subjects whose does not have a complete condition.
    # Get size of each sublist. Specifically, get the number conditions available in each of subject list
    nsize_unique_ncondition=np.array(list(map(len, alist_arr)))
    
    # Get the maximum condition being considered
    max_condition=np.max(nsize_unique_ncondition)
    
    ndex=np.where(nsize_unique_ncondition==max_condition)[0]
    # Extract list that have condition equal `max_condition`
    # consensus_partition_slice_condition_sbj= list(map(alist_arr.__getitem__,ndex))
    
    return list(map(alist_arr.__getitem__,ndex)),max_condition
def between_list_diff_index (mset, sub_set, output='retain_index'):
    '''

    Parameters
    ----------
    mset
    sub_set
    output

    Returns
    -------
    ch_names=['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FT7', 'FC3', 'FCZ', 'FC4', 'FT8', 'T3', 'C3', 'CZ', 'C4', 'T4',
    'TP7', 'CP3', 'CPZ', 'CP4', 'TP8', 'T5', 'P3', 'PZ', 'P4', 'T6', 'O1', 'OZ', 'O2']

    ch_retain=['F3', 'FZ', 'F4', 'F8', 'FC3', 'FCZ', 'FC4', 'FT8', 'O1', 'OZ', 'O2']

    retain_idx=between_list_diff_index (ch_names, ch_retain, output='retain_index')
    drop_idx=between_list_diff_index (ch_names, ch_retain, output='drop_index')

    from operator import itemgetter
    sel=list(itemgetter(*idx)(mset)) # visualise asset that has been selected
    '''

    if output == 'retain_index':
        idx = np.where ( np.in1d ( mset, sub_set ) ) [0]
    elif output == 'drop_index':
        idx = np.where ( ~np.in1d ( mset, sub_set ) ) [0]
    else:
        raise ('Please select what output you desire?')
    # convert to list
    # idx=idx.tolist ()

    return idx


def slice_array_with_list (arr, idx):
    '''
    Example
    import numpy as np
    np.random.seed ( 0 )
    arr = np.random.randint ( 0, 100, size=(2, 5, 5) )
    idx = [1, 3, 4]
    
    slice_array_with_list (arr, idx)
    '''

    ixgrid = np.ix_ ( idx, idx )
    return arr [:, ixgrid [0], ixgrid [1]]


def mesh_cor (x_max, y_max, **kwargs):
    x_min, y_min, z_min, w_min = 0, 0, 0, 0

    w_max = kwargs.get ( 'w_max', None )
    w_axis = np.arange ( w_min, w_max, dtype='int32' ) if 'w_max' in kwargs else None

    z_max = kwargs.get ( 'z_max', None )
    z_axis = np.arange ( z_min, z_max, dtype='int32' ) if 'z_max' in kwargs else None

    x_axis = np.arange ( x_min, x_max, dtype='int32' )
    y_axis = np.arange ( y_min, y_max, dtype='int32' )

    if w_axis is not None:
        mesh_coor = np.vstack ( np.meshgrid ( y_axis, x_axis, z_axis, w_axis ) ).reshape ( 4, -1 ).T
    elif z_axis is not None:
        mesh_coor = np.vstack ( np.meshgrid ( y_axis, x_axis, z_axis ) ).reshape ( 3, -1 ).T
    else:
        mesh_coor = np.vstack ( np.meshgrid ( y_axis, x_axis ) ).reshape ( 2, -1 ).T

    # swap and output is # output: method,band
    mesh_coor [:, [1, 0]] = mesh_coor [:, [0, 1]]
    return mesh_coor
