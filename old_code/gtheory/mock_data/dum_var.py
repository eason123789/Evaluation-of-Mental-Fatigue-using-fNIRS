import numpy as np
import pandas as pd
np.random.seed(0)

def md_leiden_parameter_sweep_comm_measure_extraction():
  nrep=3
  nnodes,ntime,nepoch,nsbj=30,4,10,4
  opt_k=[pd.DataFrame(np.random.randint(100,size=(nnodes,ntime))) for _ in range (nrep)]
  Qval=[np.random.rand() for _ in range(nrep)]
  return opt_k,Qval

def get_lower(a) :
  return np.tril(a,k=0)

def md_parasweep_leiden():
    # To avoid confusion, we use the term nepoch_ntime for parameter sweeping procedure. Specifically, we will take about
    # 150 slices which equivalent to 300s ~ 3min long data (assuming each slice is of length 2s). This slice start from
    # the start of the recording
    nnodes,nepoch_ntime,nsbj=3,4,2
    nsbj_arr=[np.array([get_lower(np.random.rand(nnodes,nnodes)) for _ in range(nepoch_ntime)]) for _ in range (nsbj)]
    
    return nsbj_arr
    
  
def md_modularity_fragmentation():

  nnodes,ntime,nepoch,nsbj=30,4,10,4
  epoch_com_per_sbj=[np.random.randint(np.random.randint(100,size=1),size=(nepoch,ntime,nnodes)) for _ in range(nsbj)]
  rt_local_gr=[np.random.rand(nepoch,2) for _ in range(nsbj)]
  
  return epoch_com_per_sbj, rt_local_gr

def md_flexible_clubs():
  from old_code.gtheory.utils import load_json
  config=load_json('../config.json')
  # config=load_json('config.json')
  ncondition=2

  nnodes,ntime,nepoch,nsbj=30,4,10,4
  # Example of nested list list_arr
  list_arr=[[[np.concatenate([[idx_sbj],[ncondi],[nepoch] ,np.random.rand(nnodes)]) for nepoch in range(np.random.randint(5))]\
    for ncondi in range(ncondition)] for idx_sbj in range(nsbj)]
  
  
  eeg_label=config['eloc']['eeg_list']
  nlabel=['sbj_id', 'mental_condition','epo_idx']+eeg_label
  opt_df=pd.DataFrame(np.vstack([np.vstack(x) for x in np.array(list_arr,object).ravel().tolist() if x]),columns=nlabel)

  return opt_df