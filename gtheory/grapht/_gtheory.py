import bct
import igraph as ig
import leidenalg as la
import numpy as np
import scipy.sparse
from bct.algorithms.clustering import agreement
from bct.utils import get_rng
from sklearn.metrics.cluster import adjusted_mutual_info_score,adjusted_rand_score
from sklearn.metrics import jaccard_score
from sklearn.utils.validation import check_random_state
import warnings
from gtheory.grapht.ncat import consensus_similarity
from gtheory.grapht.ncat import flexibility

def _dummyvar(labels):
    """
    Generates dummy-coded array from provided community assignment `labels`
    Parameters
    ----------
    labels : (N,) array_like
        Labels assigning `N` samples to `G` groups
    Returns
    -------
    ci : (N, G) numpy.ndarray
        Dummy-coded array where 1 indicates that a sample belongs to a group
    """

    comms = np.unique(labels)

    ci = np.zeros((len(labels), len(comms)))
    for n, grp in enumerate(comms):
        ci[:, n] = labels == grp

    return ci


def zrand_netneuro(X, Y):
    """
    Calculates the z-Rand index of two community assignments
    Parameters
    ----------
    X, Y : (n, 1) array_like
        Community assignment vectors to compare
    Returns
    -------
    z_rand : float
        Z-rand index
    References
    ----------
    Amanda L. Traud, Eric D. Kelsic, Peter J. Mucha, and Mason A. Porter.
    (2011). Comparing Community Structure to Characteristics in Online
    Collegiate Social Networks. SIAM Review, 53, 526-543.
    """

    if X.ndim > 1 or Y.ndim > 1:
        if X.shape[-1] > 1 or Y.shape[-1] > 1:
            raise ValueError('X and Y must have only one-dimension each. '
                             'Please check inputs.')

    Xf = X.flatten()
    Yf = Y.flatten()

    n = len(Xf)
    indx, indy = _dummyvar(Xf), _dummyvar(Yf)
    Xa = indx.dot(indx.T)
    Ya = indy.dot(indy.T)

    M = n * (n - 1) / 2
    M1 = Xa.nonzero()[0].size / 2
    M2 = Ya.nonzero()[0].size / 2

    wab = np.logical_and(Xa, Ya).nonzero()[0].size / 2

    mod = n * (n**2 - 3 * n - 2)
    C1 = mod - (8 * (n + 1) * M1) + (4 * np.power(indx.sum(0), 3).sum())
    C2 = mod - (8 * (n + 1) * M2) + (4 * np.power(indy.sum(0), 3).sum())

    a = M / 16
    b = ((4 * M1 - 2 * M)**2) * ((4 * M2 - 2 * M)**2) / (256 * (M**2))
    c = C1 * C2 / (16 * n * (n - 1) * (n - 2))
    d = ((((4 * M1 - 2 * M)**2) - (4 * C1) - (4 * M))
         * (((4 * M2 - 2 * M)**2) - (4 * C2) - (4 * M))
         / (64 * n * (n - 1) * (n - 2) * (n - 3)))

    sigw2 = a - b + c + d
    # catch any negatives
    if sigw2 < 0:
        return 0
    z_rand = (wab - ((M1 * M2) / M)) / np.sqrt(sigw2)

    return z_rand



def consensus_parti_params(consensus_partition, max_zrandval,Q_avg_npart,nsize_nodes_windows):
  
  # Get consensus partition flexibility
  artt=np.reshape(consensus_partition, list(reversed(nsize_nodes_windows))).T
  flex_cp_a=flexibilities_arr(artt)
  flex_cp=np.mean(flex_cp_a)
  
  # Get the consensus partition community number
  ncommunities_cp=np.unique(consensus_partition).shape[0]
  
  # zrand average over N optimization: add by rpb
  
  # Calculate and extract the Q multiply Zrand measure of the representative parti
  q_zrand=max_zrandval*Q_avg_npart
  
  return flex_cp,ncommunities_cp,q_zrand


# def get_con_parti_var(C,average_pairwise_simm,Qval,nsize_nodes_windows):
#   """
#   Get average_pairwise_simm having the maximum values. This show it can represent the other N-1 solutions
#
#   """
#
#   (max_zrandval,idx_pos) = (np.max(average_pairwise_simm), np.argmax(average_pairwise_simm))
#   consensus_partition = C[idx_pos,:]
#   # consensus_simm_idx = X
#   flex_cp,ncommunities_cp,q_zrand=consensus_parti_params(consensus_partition, max_zrandval,Qval,
#     nsize_nodes_windows)
#   #TODOss
#   # warnings.warn("Most probably skip reshaping and let it be as later under the amis_cpartition=ars_cpartition.reshape("
#   #               "ashape[0],ashape[1],ashape[2]*ashape[3],order='F'), which will result in array of shape nsbj_epoch x omega_gamma_pair, node*slices")
#   # consensus_partition=consensus_partition.reshape(-1,nnodes_size).T
#   return (consensus_partition,flex_cp,ncommunities_cp,q_zrand)

def get_sub_zrand(C,nsize_nodes_windows=None,Qval=None,con_parti=False,nshape_arr=None,zrand_opt=False):
  '''
  Inputs:     C,      pxn matrix of community assignments where p is the
                        number of optimizations and n the number of nodes
  Returns
  -------
   Used by
   similarity_between_subject()
   
   Similarity score between -1.0 and 1.0. Random labelings have an ARI close to 0.0. 1.0 stands for perfect match.
  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html
  '''
  
  # Number of subjects
  npart = len(C[:,0])
  # Initialize variables
  
  pairwise_simm = np.zeros(shape=(npart,npart), dtype = np.double)

  for i in range(npart):
    for j in range(i+1,npart):
      pairwise_simm[i,j]=zrand_netneuro(C[i,:],C[j,:])

  pairwise_simm = pairwise_simm + pairwise_simm.T


  # Average pairwise similarity. Average similarity solution_x to other solution_non_x
  average_pairwise_simm = np.sum(pairwise_simm ,axis = 1)/(npart-1)
  
  if zrand_opt:
      ## For time being, I have verified this output is used for 5_omega_lambda space where I want the zrand average
      # outp. SImilarity between subject
      # g=np.mean(average_pairwise_simm)
      return np.mean(average_pairwise_simm)
  
  
  # Deliberately insert another function after the if statement above
  (max_zrandval,idx_pos) = (np.max(average_pairwise_simm), np.argmax(average_pairwise_simm))
  consensus_partition = C[idx_pos,:]
  # consensus_simm_idx = X



  
  if con_parti:

    flex_cp, ncommunities_cp, q_zrand = consensus_parti_params(consensus_partition, max_zrandval,
                                                               Qval,
                                                               nsize_nodes_windows)

    # cpartition,flex_cp,ncommunities_cp,q_zrand=get_con_parti_var(C,
    #   average_pairwise_simm,Qval,nsize_nodes_windows)
    
    all_val=dict(flex_cp=flex_cp,ncommunities_cp=ncommunities_cp,q_zrand=q_zrand,cpartition={"cpartition":consensus_partition})
    return all_val
  
  elif isinstance(nshape_arr, tuple):
    # Used in _rep_partition_psols_pepoch. 6a make_subject_condition_average_presentation_leidean.py
      return consensus_partition.reshape(list(reversed(nshape_arr)))



def comm_measure_louvain(opt_k, Q, consensus_partition_opt=True):
  '''
  The format for opt_k and Q as derived from
  md_leiden_parameter_sweep_comm_measure_extraction()
  from gtheory.grapht import flexibilities_arr
  :param opt_k:
  :param Q:
  :param consensus_partition_opt:
  :return:
  '''
  Q_avg = sum(Q) / len(Q)
  
  # get a N_Optimization-flexibility average by the sum(flex_avg_npart) / len(flex_avg_npart)
  flex_avg_npart = [flexibilities_arr(df_idx) for df_idx in opt_k]
  
  # return a single flexibility number: Flexibility of all nodes
  flex_avg_npart = np.array(flex_avg_npart).mean(axis=0).mean()
  
  # Number of unique communities accross the slices over all optimization numbers
  ncommunities = np.unique(np.array(opt_k)).shape[0]
  
  # Get the comms measure from the consensus partition
  # First extract the consensus partition out of all the optimisations
  
  # >> Slow approach
  # b_arr=np.array([narr.T.reshape(-1) for narr in np.array(opt_k)])
  
  # Efficient numpy approach
  # https://stackoverflow.com/a/70085398/6446053
  # b_arr = np.reshape(np.array(opt_k).T, (np.prod(np.array(opt_k[0].shape)) , -1)).T
  b_arr = np.array(opt_k).reshape(len(opt_k), np.prod(np.array(opt_k[0].shape)), order='F')
  consensus_partition, consensus_simm_idx, _, zrand_average_npart = consensus_similarity(b_arr)
  
  # Get consensus partition flexibility
  artt = np.reshape(consensus_partition, list(reversed(opt_k[0].shape))).T
  flex_cp_a = flexibilities_arr(artt)
  flex_cp = np.mean(flex_cp_a)
  
  # Get the consensus partition community number
  ncommunities_cp = np.unique(consensus_partition).shape[0]
  
  # Calculate and extract the Q_Zrand measure
  q_zrand = consensus_simm_idx * Q_avg
  
  if consensus_partition_opt:
    # This output type used for parameter sweeping
    return Q_avg, flex_avg_npart, ncommunities, flex_cp, ncommunities_cp, consensus_partition, q_zrand
  
  # This output type used for statistical analysis. and, consensus partition is not require
  return Q_avg, flex_avg_npart, ncommunities, flex_cp, ncommunities_cp, q_zrand


def comm_measure_leiden(opt_k,Q,nnodes_size=None) :
  '''
  The format for opt_k and Q as derived from
  md_leiden_parameter_sweep_comm_measure_extraction()
  from gtheory.grapht import flexibilities_arr
  :param opt_k:
  :param Q:
  :param consensus_partition_opt:
  :return:
  '''
  nsize_nodes_windows=opt_k[0].shape
  Q_avg_npart=sum(Q)/len(Q)
  
  # get a N_Optimization-flexibility average by the sum(flex_avg_npart) / len(flex_avg_npart)
  flex_avg_npart=[ flexibilities_arr(df_idx) for df_idx in opt_k]
  
  # return a single flexibility number: Flexibility of all nodes
  flex_avg_npart=np.array(flex_avg_npart).mean(axis=0).mean()
  
  # Number of unique communities accross the slices over all optimization numbers
  # HH=np.array(opt_k)
  ncommunities=sum([np.unique(arrr).shape[0] for arrr in np.array(opt_k)])/len(opt_k)
  
  
  all_sol_var=dict(Q_avg_npart=Q_avg_npart,flex_avg_npart=flex_avg_npart,ncommunities=ncommunities)
  
  # Get the comms measure from the consensus partition
  # First extract the consensus partition out of all the optimisations
  
  
  # Efficient numpy approach
  # https://stackoverflow.com/a/70085398/6446053
  # b_arr = np.reshape(np.array(opt_k).T, (np.prod(np.array(opt_k[0].shape)) , -1)).T
  b_arr=np.array(opt_k).reshape(len(opt_k),np.prod(np.array(opt_k[0].shape)),order='F')
  
  # Previously known as
  # consensus_partition,consensus_simm_idx,_,zrand_average_npart=consensus_similarity(b_arr)
  
  ## Change to
  cpartition_res=get_sub_zrand(b_arr,nsize_nodes_windows=nsize_nodes_windows,
                                Qval=Q_avg_npart,con_parti=True)
  
  all_sol_var.update(cpartition_res)
  
  

  # This output type used for parameter sweeping
  return all_sol_var
  
  # # This output type used for statistical analysis. and, consensus partition is not require
  # return Q_avg_npart,flex_avg_npart,ncommunities,flex_cp,ncommunities_cp,q_zrand


def flexibilities_arr(df):
  '''
  FLEXIBILITY    Flexibility coefficient
  F = FLEXIBILITY(S) calculates the flexibility coefficient of
  S. The flexibility of each node corresponds to the number of times that
  it changes module allegiance, normalized by the total possible number
  of changes.
  
  Currently we only accept temporal network:
  In temporal networks, we consider changes possible only between adjacent
  time points.
  
  In multislice/categorical networks, module allegiance changes are possible
  between any pairs of slices (KIV).
  Inputs:     S,      nxp matrix of community assignments where p is the
                     number of slices/layers and n the number of nodes
          nettype,   string specifying the type of the network:
                     'temp'  temporal network (default)
                     'cat'   categorical network
  Outputs:    F,      Flexibility coefficient
  '''
  
  # temporal network implementation
  if isinstance(df, np.ndarray):
    possibleChanges = df.shape[1]-1
    F=(np.diff(df,axis=1) != 0).sum(axis=1)/possibleChanges
  else:
    possibleChanges = len(df.columns)-1
    F=(np.diff(df.values,axis=1) != 0).sum(axis=1)/possibleChanges
  return F


def combine_ncommunities_sbj_epoch_identity(sbj_idx,rt_local_sbj,ndarr) :
  nsize=ndarr.shape[0]
  arr_info=np.array(([sbj_idx]*nsize,list(range(nsize)))).transpose()
  return np.concatenate((arr_info,rt_local_sbj,ndarr),axis=1)

def sorting(a):
  b = np.sort(a,axis=1)
  return (b[:,1:] != b[:,:-1]).sum(axis=1)+1

def ncommunities_slices(nepoch) :
  '''
  
  :param nepoch: Is a communities of shape (n,p)
  :return: number of communities at each slice and accross all slices. The last columns always the unique communities
  accross all slices
 
  '''
  # nepochs=nepoch.T
  # Unique communities accross the all time slices
  unique_comm=np.unique(nepoch)
  
  # Unique communities at each slice
  # https://stackoverflow.com/a/51308486/6446053
  ncomm_per_slice=np.count_nonzero(np.diff(np.sort(nepoch.T)), axis=1)+1
  # comm=np.concatenate((ncomm_per_slice,np.array([unique_comm.shape[0]])))
  return np.concatenate((ncomm_per_slice,np.array([unique_comm.shape[0]])))


def modular_allegiance(a) :
  '''
    nodal_association_matrices_python
    https://stackoverflow.com/a/69315800/6446053
    
    This is updated Python version and has been compared with the original matlab version as shown in the link
    :param a:
    :return:
    '''
  return (a.T==a.T[:,None]).sum(2).astype(np.double)


def leiden_single_slice(agreement,gamma,n_iterations=-1) :
  # G = ig.Graph.Weighted_Adjacency (agreement)
  # G.vs ['name'] = nlabel
  partition=la.find_partition(ig.Graph.Weighted_Adjacency(agreement),la.CPMVertexPartition,resolution_parameter=gamma,n_iterations=n_iterations)
  # bb=np.array(partition.membership)
  # bb=np.array(partition.membership).reshape((-1,1))
  # n=1
  # ig.plot(G,vertex_label = [f'{v["name"]}' for v in G.vs])
  # ig.plot(partition,vertex_label = [f'{v["name"]}' for v in G.vs])
  return np.array(partition.membership)


def rpb_consensus_und(D,tau,algo_type,reps=1000,seed=None,gamma_rpb=1) :
  '''
    RPB modify using leiden algorithm
    Copy from: https://github.com/aestrivex/bctpy/blob/32c7fe7345b281c2d4e184f5379c425c36f3bbc7/bct/algorithms/clustering.py#L353

  This algorithm seeks a consensus partition of the
  agreement matrix D. The algorithm used here is almost identical to the
  one introduced in Lancichinetti & Fortunato (2012): The agreement
  matrix D is thresholded at a level TAU to remove an weak elements. The
  resulting matrix is then partitions REPS number of times using the
  Louvain algorithm (in principle, any clustering algorithm that can
  handle weighted matrixes is a suitable alternative to the Louvain
  algorithm and can be substituted in its place). This clustering
  produces a set of partitions from which a new agreement is built. If
  the partitions have not converged to a single representative partition,
  the above process repeats itself, starting with the newly built
  agreement matrix.
  NOTE: In this implementation, the elements of the agreement matrix must
  be converted into probabilities.
  NOTE: This implementation is slightly different from the original
  algorithm proposed by Lanchichinetti & Fortunato. In its original
  version, if the thresholding produces singleton communities, those
  nodes are reconnected to the network. Here, we leave any singleton
  communities disconnected.
  Parameters
  ----------
  D : NxN np.ndarray
      agreement matrix with entries between 0 and 1 denoting the probability
      of finding node i in the same cluster as node j
  tau : float
      threshold which controls the resolution of the reclustering
  reps : int
      number of times the clustering algorithm is reapplied. default value
      is 1000.
  seed : hashable, optional
      If None (default), use the np.random's global random state to generate random numbers.
      Otherwise, use a new np.random.RandomState instance seeded with the given value.
  Returns
  -------
  ciu : Nx1 np.ndarray
      consensus partition
  '''
  rng=get_rng(seed)
  
  def unique_partitions(cis) :
    # relabels the partitions to recognize different numbers on same
    # topology
    
    n,r=np.shape(cis)  # ci represents one vector for each rep
    ci_tmp=np.zeros(n)
    
    for i in range(r) :
      for j,u in enumerate(sorted(np.unique(cis[:,i],return_index=True)[1])) :
        ci_tmp[np.where(cis[:,i]==cis[u,i])]=j
      cis[:,i]=ci_tmp  # so far no partitions have been deleted from ci
    
    # now squash any of the partitions that are completely identical
    # do not delete them from ci which needs to stay same size, so make
    # copy
    ciu=[]
    cis=cis.copy()
    c=np.arange(r)
    # count=0
    while (c!=0).sum()>0 :
      ciu.append(cis[:,0])
      dup=np.where(np.sum(np.abs(cis.T-cis[:,0]),axis=1)==0)
      cis=np.delete(cis,dup,axis=1)
      c=np.delete(c,dup)  # count+=1  # print count,c,dup  # if count>10:  #	class QualitativeError(): pass  #	raise QualitativeError()
    return np.transpose(ciu)
  
  n=len(D)
  flag=True
  while flag :
    flag=False
    dt=D*(D>=tau)
    np.fill_diagonal(dt,0)
    
    if np.size(np.where(dt==0))==0 :
      ciu=np.arange(1,n+1)
    else :
      cis=np.zeros((n,reps))
      for i in np.arange(reps) :
        # cis[:, i], _ = modularity_louvain_und_sign(dt, seed=rng)
        # cis[:, i], _ = bct.algorithms.modularity_louvain_und_sign(dt, gamma=0.1, seed=rng)
        if algo_type=='louvain':
          cis[:, i], _ = bct.algorithms.modularity_louvain_und_sign(dt, gamma=gamma_rpb, seed=rng)
        elif algo_type=='leiden':
          cis[:,i]=leiden_single_slice(dt,gamma_rpb,n_iterations=-1)
      ciu=unique_partitions(cis)
      nu=np.size(ciu,axis=1)
      if nu>1 :
        flag=True
        D=agreement(cis)/reps
  
  return np.squeeze(ciu+1)


def find_consensus(assignments,algo_type,null_func=np.mean,return_agreement=False,seed=None,
 gamma=1,nrep=100) :
  """
    Copy from:
    https://github.com/netneurolab/netneurotools/blob/503d9ade5af5531207b4b623816dd9bf4ac3e235/netneurotools/cluster.py#L317

    Tweak: Previously, the function does not have the flexibility to select diffrent gamma. A standard gamma equal to 1
    was used. In this revision, we allow the function `find_consensus` to accept diffrent gamma value via the
    variable `gamma`
    Finds consensus clustering labels from cluster solutions in `assignments`
    Parameters
    ----------
    assignments : (N, M) array_like
        Array of `M` clustering solutions for `N` samples (e.g., subjects,
        nodes, etc). Values of array should be integer-based cluster assignment
        labels
    null_func : callable, optional
        Function used to generate null model when performing consensus-based
        clustering. Must accept a 2D array as input and return a single value.
        Default: :func:`numpy.mean`
    return_agreement : bool, optional
        Whether to return the thresholded N x N agreement matrix used in
        generating the final consensus clustering solution. Default: False
    seed : {int, np.random.RandomState instance, None}, optional
        Seed for random number generation. Used when permuting cluster
        assignments during generation of null model. Default: None
    Returns
    -------
    consensus : (N,) numpy.ndarray
        Consensus cluster labels
    References
    ----------
    Bassett, D. S., Porter, M. A., Wymbs, N. F., Grafton, S. T., Carlson,
    J. M., & Mucha, P. J. (2013). Robust detection of dynamic community
    structure in networks. Chaos: An Interdisciplinary Journal of Nonlinear
    Science, 23(1), 013142.
    """
  
  rs=check_random_state(seed)
  samp,comm=assignments.shape
  
  # create agreement matrix from input community assignments and convert to
  # probability matrix by dividing by `comm`
  agreement=bct.clustering.agreement(assignments,buffsz=samp)/comm
  
  # generate null agreement matrix and use to create threshold
  null_assign=np.column_stack([rs.permutation(i) for i in assignments.T])
  null_agree=bct.clustering.agreement(null_assign,buffsz=samp)/comm
  threshold=null_func(null_agree)
  # np.savez('dev_consensus_part_leiden', agreement=agreement, threshold=threshold)
  # run consensus clustering on agreement matrix after thresholding
  # consensus = bct.clustering.consensus_und(agreement, threshold, 10) # Originally other than rpb utilise this line
  consensus=rpb_consensus_und(agreement,threshold,algo_type,nrep,gamma_rpb=gamma)
  
  if return_agreement :
    return consensus.astype(int),agreement*(agreement>threshold)
  
  return consensus.astype(int)


def _flexibility_average_npartition(b_arr,T,N,npar) :
  # Iterate through each optimization (100x). Since previously we obtained 1D numpy, we need to covert to slice x nodes
  # shapes as a suitable input into the flexibility function. Flexibility will return array of shape slice x
  # probability of node changes. The rows is redundent. To get average nodes flexibility, get the columns average.
  # Ultimately, we going to have a list of average node flexibility, with the list has a len to N_Optimization. Then,
  # get a N_Optimization-flexibility average by the sum(flex_avg_npart) / len(flex_avg_npart)
  
  return [np.mean(flexibility(np.reshape(b_arr[idx,:],[T,N])),axis=1)[0] for idx in range(npar)]

# def comm_measure_leiden(arr,Q,T,N,npar,consensus_partition_opt=True) :
#   '''
#
#       Parameters
#       ----------
#       T
#       N
#       npar
#
#       Returns
#       -------
#
#       '''
#   # T, N=4,5
#   # npar=3
#   # arr=[np.random.randint (8, size=20) for _ in range(npar)]
#   b_arr=np.array(arr)
#
#   Q_avg=sum(Q)/len(Q)
#   # Q_avg_tmu=Q_avg/twomu
#
#   # get a N_Optimization-flexibility average by the sum(flex_avg_npart) / len(flex_avg_npart)
#   flex_avg_npart=_flexibility_average_npartition(b_arr,T,N,npar)
#   flex_avg_npart=sum(flex_avg_npart)/len(flex_avg_npart)
#
#   ncommunities=np.unique(b_arr).shape[0]
#
#   # Get the comms measure from the consensus partition
#   # First extract the consensus partition out of all the optimisations
#   consensus_partition,consensus_simm_idx,_,zrand_average_npart=consensus_similarity(b_arr)
#
#   # Get consensus partition flexibility
#   flex_cp=np.mean(flexibility(np.reshape(consensus_partition,[T,N])),axis=1)[0]
#
#   # Get the consensus partition community number
#   ncommunities_cp=np.unique(consensus_partition).shape[0]
#
#   # Calculate and extract the Q_Zrand measure
#   q_zrand=consensus_simm_idx*Q_avg_tmu
#
#   if consensus_partition_opt :
#     # This output type used for parameter sweeping
#     return Q_avg_tmu,flex_avg_npart,ncommunities,flex_cp,ncommunities_cp,consensus_partition,q_zrand
#
#   # This output type used for statistical analysis. ANd, consensus partition is not require
#   return Q_avg_tmu,flex_avg_npart,ncommunities,flex_cp,ncommunities_cp,q_zrand

def comm_measure(arr,Q,twomu,T,N,npar,consensus_partition_opt=True) :
  '''
      
      Parameters
      ----------
      T
      N
      npar
  
      Returns
      -------
  
      '''
  # T, N=4,5
  # npar=3
  # arr=[np.random.randint (8, size=20) for _ in range(npar)]
  b_arr=np.array(arr)
  
  Q_avg=sum(Q)/len(Q)
  Q_avg_tmu=Q_avg/twomu
  
  # get a N_Optimization-flexibility average by the sum(flex_avg_npart) / len(flex_avg_npart)
  flex_avg_npart=_flexibility_average_npartition(b_arr,T,N,npar)
  flex_avg_npart=sum(flex_avg_npart)/len(flex_avg_npart)
  
  ncommunities=np.unique(b_arr).shape[0]
  
  # Get the comms measure from the consensus partition
  # First extract the consensus partition out of all the optimisations
  consensus_partition,consensus_simm_idx,_,zrand_average_npart=consensus_similarity(b_arr)
  
  # Get consensus partition flexibility
  flex_cp=np.mean(flexibility(np.reshape(consensus_partition,[T,N])),axis=1)[0]
  
  # Get the consensus partition community number
  ncommunities_cp=np.unique(consensus_partition).shape[0]
  
  # Calculate and extract the Q_Zrand measure
  q_zrand=consensus_simm_idx*Q_avg_tmu
  
  if consensus_partition_opt :
    # This output type used for parameter sweeping
    return Q_avg_tmu,flex_avg_npart,ncommunities,flex_cp,ncommunities_cp,consensus_partition,q_zrand
  
  # This output type used for statistical analysis. ANd, consensus partition is not require
  return Q_avg_tmu,flex_avg_npart,ncommunities,flex_cp,ncommunities_cp,q_zrand


def supra_adj_matrix_list(A,gamma,omega) :
  '''
    
    Here we represent intralayer in a single "supra-adjacency"
    Original version extracted from
    https://github.com/nangongwubu/Python-Version-for-Network-Community-Architecture-Toobox/blob/master/build/lib/test.py
    Parameters
    ----------
    A
    gamma
    omega

    Returns
    -------
    '''
  #
  
  N=A.shape[0]  # Number of Nodes
  T=A.shape[2]  # Number of slice
  
  B=scipy.sparse.csr_matrix((N*T,N*T))
  twomu=0
  for s in range(T) :
    k=np.sum(A[:,:,s],axis=0)
    twom=np.sum(k)
    twomu=twomu+twom
    indx=np.array(range(N))+(s*N)
    B[indx[0] :indx[-1]+1,indx[0] :indx[-1]+1]=np.subtract(A[:,:,s],(gamma*np.asmatrix(k).T*np.asmatrix(k)/twom))
  
  twomu=twomu+2*omega*N*(T-1)
  
  solve=scipy.sparse.spdiags(np.ones((N*T,2)).T,np.array([-N,N]),N*T,T*N)
  addthis=omega*solve
  # daddthis = addthis.todense()
  # B = B.todense() + daddthis
  
  return B.todense()+addthis.todense(),twomu


def most_common(arr) :
  '''
    
    Return an array of the modal (most common) value in the passed array.

    If there is more than one such value, only the smallest is returned. The bin-count for the modal bins is also returned.
    Parameters
    ----------
    arr

    Returns
    -------

    '''
  # https://stackoverflow.com/q/16330831/6446053
  from scipy.stats import mode
  # arr=np.array([[0,0,0,0,0],[0,4,1,1,1],[0,1,1,2,2],[0,3,2,2,2]])
  counts=mode(arr,axis=1)
  # print(counts[0])
  return counts[0]


def get_community_each_epoch(sbj1_path) :
  '''
    This is only temporary
    Returns
    -------

    '''
  from tqdm import tqdm
  from os.path import (split)
  import os.path
  import pickle
  import numpy as np
  from gtheory.grapht import gt_leidenalg
  from gtheory.utils import between_list_diff_index
  from gtheory.utils.nparray import slice_array_with_list
  root_folder,fname=split(sbj1_path)
  fsave=os.path.join(root_folder,'community_theta.pickle')
  arr=np.load(sbj1_path)
  con=arr['con_arr']  # (epochs, nwindows, spectral_method, nchan, nchan, fbands)
  ch_names=arr['ch_name'].tolist()
  nepochs=con.shape[0]
  # EXTRACT SOME
  
  # epoch_idx = 0
  spec_method_idx=0
  fband_idx=0
  all_com=[]
  for nepoch_idx in tqdm(range(nepochs)) :
    arr_temporal=con[nepoch_idx,:,spec_method_idx,:,:,fband_idx]
    arr_temporal=np.where(arr_temporal<0.8,0,arr_temporal)
    
    ch_retain=['F3','FZ','F4','F8','FC3','FCZ','FC4','FT8','O1','OZ','O2']
    retain_idx=between_list_diff_index(ch_names,ch_retain,output='retain_index')
    arr_temporal=slice_array_with_list(arr_temporal,retain_idx.tolist())
    ch_names=ch_retain
    
    nslice=arr_temporal.shape[0]
    slice_name=[f'Slice {idx}' for idx in range(nslice)]
    gtl=gt_leidenalg(arr_temporal,ch_names,slice_name=slice_name)
    # gtl.viz_membership_temporal ()
    communities=gtl.arr_to_membership()
    communities=communities.values
    all_com.append(communities)  # n=1
  with open(fsave,'wb') as handle :
    pickle.dump(all_com,handle,protocol=pickle.HIGHEST_PROTOCOL)


def flexibility_consec_time_windows(communities) :
  '''

    Returns
    -------
    df = pd.DataFrame ( [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 2]],
                        columns=['slice 1', 'slice 2', 'slice 3'],
                        index=['A', 'B', 'C', 'D'] )
    communities = df.values
    # The communities nslice x nodes
    communities = df.values.T
    nslice=communities.shape[0]
    pair_low, pair_top = np.arange(0, nslice-1, 1),np.arange(1, nslice, 1)
    mean_node=[np.mean(flexibility(communities[(x,y),:])[0,:]) for x,y in zip(pair_low,pair_top)]

    '''
  # The communities nslice x nodes
  # communities = df.values.T
  #### Method if using the toolbox
  # method_use='ncat'
  method_use='np'
  # if method_use=='ncat':
  nslice=communities.shape[0]
  pair_low,pair_top=np.arange(0,nslice-1,1),np.arange(1,nslice,1)
  
  mean_node=[np.mean(flexibility(communities[(x,y),:])[0,:]) for x,y in zip(pair_low,pair_top)]
  
  # else:
  # communities = np.array([[3,1,0,4,10],
  #           [1,-1,0,4,11],
  #           [1,0,0,5,11],
  #           [1,1,0,5,13]])
  k=communities.shape[1]
  nres=np.invert(np.diff(communities,axis=0)==0)  # << take the difference between the each row and the one above it
  
  nmean=nres.sum(axis=1)/k
  # kk=np.array(mean_node)
  if not np.array_equal(np.array(mean_node),nmean) :
    raise ('Numpy shortcut is not same with ncat approach. COnsider use ncat only')
  # if
  return nmean


# https://teneto.readthedocs.io/en/latest/
def flexibility_index(x) :
  """ Flexibility Index
    In the context of graph clustering it was defined in (Basset2011_), flexbility is the frequency
    of a nodea change module allegiance; the transition of brain states between consecutive
    temporal segments. The higher the number of changes, the larger the FI will be.
    .. math::
       FI = \\frac{\\text{number of transitions}} {\\text{total symbols - 1}}
    |
    .. [Basset2011] Bassett, D. S., Wymbs, N. F., Porter, M. A., Mucha, P. J., Carlson, J. M., & Grafton, S. T. (2011). Dynamic reconfiguration of human brain networks during learning. Proceedings of the National Academy of Sciences, 108(18), 7641-7646.



    Compute the flexibility index for the given symbolic, 1d time series.
    Parameters
    ----------
    x : array-like, shape(N)
        Input symbolic time series.
    Returns
    -------
    fi : float
        The flexibility index.
    """
  l=len(x)
  
  counter=0
  for k in range(l-1) :
    if x[k]!=x[k+1] :
      counter+=1
  
  fi=counter/np.float32(l-1)
  
  return fi


def assign_label(G,label) :
  G.vs['id']=label
  return G


def time_slices_to_layers(G_ls,interslice_weight) :
  G_layers,G_interslice,G=la.time_slices_to_layers(G_ls,interslice_weight=interslice_weight,slice_attr='slice',vertex_id_attr='id',edge_type_attr='type',weight_attr='weight')
  return G_layers,G_interslice,G


def _community_membership(G_ls,resolution_parameter=None,interslice_weight=None) :
  '''

    Gamma:resolution_parameter 3
    Omega:interslice_weight  1e-1
    :param G_ls:
    :param gamma:
    :param interslice_weight:
    :return:
    '''
  
  G_layers,G_interslice,G=time_slices_to_layers(G_ls,interslice_weight)
  # Create partitions
  
  partitions=[la.CPMVertexPartition(H,node_sizes='node_size',weights='weight',resolution_parameter=resolution_parameter)
    for H in G_layers]
  interslice_partition=la.CPMVertexPartition(G_interslice,resolution_parameter=resolution_parameter,node_sizes='node_size',weights='weight')
  
  # Detect communities
  optimiser=la.Optimiser()
  optimiser.set_rng_seed(11)
  diff=optimiser.optimise_partition_multiplex(partitions+[interslice_partition])
  
  # Plot network
  # Note here, the`v` is assigned to `id` since we define it as id at the previous step
  partition_all=ig.VertexClustering(G,partitions[0].membership)
  
  return partition_all


def community_membership(con_per_epochs,freq_idx,ch_names,interslice_weight=None,resolution_parameter=None,niteration=1) :
  G_all=[]
  for idx_epoch,con in enumerate(con_per_epochs) :
    g_l=ig.Graph.Weighted_Adjacency(con[:,:,freq_idx].tolist())
    G_all.append(g_l)
  
  G_ls=[assign_label(G,ch_names) for G in G_all]
  
  partition=[_community_membership(G_ls,resolution_parameter=resolution_parameter,interslice_weight=interslice_weight)
    for _ in range(niteration)]
  
  _,_,G=time_slices_to_layers(G_ls,interslice_weight)
  
  # ig.plot ( partition[0], vertex_label=[f'{v ["id"]}-{v ["slice"]}' for v in G.vs], )
  return partition,G


def allegiance(community) :
  u"""
    Computes allience of communities.

    The allegiance matrix with values representing the probability that
    nodes i and j were assigned to the same community by time-varying clustering methods.[alleg-1]_

    parameters
    ----------
    community : array
        array of community assignment of size node,time

    returns
    -------
    P : array
        module allegiance matrix, with P_ij probability that area i and j are in the same community

    Reference:
    ----------

    .. [alleg-1]:

        Bassett, et al. (2013)
        “Robust detection of dynamic community structure in networks”, Chaos, 23, 1

    """
  N=community.shape[0]
  C=community.shape[1]
  T=P=np.zeros([N,N])
  
  for t in range(len(community[0,:])) :
    for i in range(len(community[:,0])) :
      for j in range(len(community[:,0])) :
        if i==j :
          continue
        # T_ij indicates the number of times that i and j are assigned to the same community across time
        if community[i][t]==community[j][t] :
          T[i,j]+=1
  
  # module allegiance matrix, probability that ij were assigned to the same community
  P=(1/C)*T
  # Make diagonal nan
  np.fill_diagonal(P,np.nan)
  return P
