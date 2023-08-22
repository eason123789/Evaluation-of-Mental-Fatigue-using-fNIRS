# -*- coding: utf-8 -*-
"""Some utility functions."""

# Authors: rpb>
#
# License: BSD (3-clause)
import os
import re

import numpy as np
import pandas as pd

def range_step(dpar,para):
  gval=dpar[para].drop_duplicates().sort_values(ascending=True).to_frame().reset_index(drop=True)
  gval['diff']=gval[para].diff()
  mval=gval['diff'].max().round(10)
  decimal_places = abs(int(f'{mval:e}'.split('e')[-1]))+3
  d=dict(step=gval['diff'].max().round(decimal_places),
    pmax=dpar[para].max().round(2),
    pmin=dpar[para].min().round(2),
    var=para)
  return d

def load_events_label(npath,pt,clustertype=None,spliter_s=None,rtime=False,df_output_type='array',**kwargs) :
  if spliter_s is None :
    spliter_s=pt['community_leiden']
  
  filter_mid= kwargs.get('filter_mid',None)
  pathlabel=re.search(rf"(.*?(?={spliter_s}))",npath).group(0)
  
  if not filter_mid:
    opath=os.path.join(pathlabel,pt['filtered_event_fn'])
  else:
    opath=os.path.join(pathlabel,pt['sel_events'])
    

  df=pd.read_pickle(opath)
  
  if clustertype is None:
    return df
  elif clustertype=='ncluster_all' :
    clustertype_all=['ncluster_2','ncluster_3','ncluster_4','ncluster_5']
    if df_output_type=='array':
      arr_label=df[clustertype_all].to_numpy()
    else:
      arr_label=df[clustertype_all]
  else :
    arr_label=df[clustertype].to_numpy()
  
  if rtime :
    rt_both=df[['rt_local_sec','rt_global_sec']]
    return arr_label,rt_both.to_numpy()
  
  return arr_label


def load_melt_df_barplot(df,discluster,var_name=None,value_name=None) :
  # df=pd.read_pickle(opath)
  ncolumn=['frontal','temporalleft','temporalright','central','occipital']
  xlabel=[discluster]+ncolumn
  ndf=df[xlabel] # check to change name
  ndf=ndf.rename(columns={discluster: 'mental_condition'})
  unique_label=ndf['mental_condition'].unique().tolist()
  # https://bit.ly/31CwObI
  df_melt=pd.melt(ndf,id_vars='mental_condition',var_name=var_name,value_name=value_name)
  # df_melt=pd.melt(ndf,id_vars='mental_condition',var_name='brain_region',value_name='flexibility')
  return df_melt,unique_label


def save_list_df_multiple_sheets(df_alist,sheet_id,spath) :
  writer=pd.ExcelWriter(spath)
  for df,nclus in zip(df_alist,sheet_id) :
    df.to_excel(writer,sheet_name=f'cluster_{nclus}')
  writer.save()


def assign_pair(arr_list,df_series,nspacing) :
  # mdf = pd.DataFrame ( ['A', 'D', 'E', 'Z'], columns=['ref'] )
  # al_ref = np.array ( [[ndex] * 3 for ndex in mdf ['ref'].values.tolist ()] ).reshape ( -1, 1 )
  # table = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
  # df = pd.DataFrame ( np.array ( table ).reshape ( -1, 1 ), columns=['nval'] )
  # df ['group'] = al_ref
  # print ( df )
  n_event=[f'{idx}' for idx in range(len(arr_list))]
  n_event_l=np.array([[ndex]*nspacing for ndex in n_event]).reshape(-1,1)
  al_ref=np.array([[ndex]*nspacing for ndex in df_series.tolist()]).reshape(-1,1)
  df=pd.DataFrame(np.array(arr_list).reshape(-1,1),columns=['nval'])
  df['group']=al_ref
  df['epoch_seq']=n_event_l
  return df
  n=1
