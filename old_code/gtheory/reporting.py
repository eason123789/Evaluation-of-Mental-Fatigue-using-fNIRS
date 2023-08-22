'''

Refer:  https://github.com/neuropycon . Interesting how to store folder and do the reporting
'''
import re
from re import findall

import pandas as pd
from tqdm import tqdm


def make_generic_report_layout(fig_list,fanal_list,spath,title=None,caption=False):
  import mne
  rep = mne.Report(title=title)
  if not caption:
    for efig,efnal in zip(fig_list,fanal_list):
      rep.add_figs_to_section(efig,captions=efnal,section=efnal)
  
  elif caption:
    """
    Used for modularity fragmentation
    """
    for disfig,disanalysis in zip(fig_list,fanal_list):
      lcaption=[f'{idx}'for idx in range(len(disfig))]
      rep.add_figs_to_section(disfig,captions=lcaption,section=disanalysis)
  
  else:
    '''
    Used for correlation plotting except for modularity fragmentation
    '''
    for disfig,disanalysis,discaption in zip(fig_list,fanal_list,caption):
      lcaption=discaption
      rep.add_figs_to_section(disfig,captions=lcaption,section=disanalysis)

  
  rep.save ( spath, overwrite=True, open_browser=False )
  
def get_cluster_per_subject (xpath, ncluster_col,psave=None):
    '''
    Produce class_subject distribution report. This give us an overview about the distribution of fatigue condtion
    for each subject and can assist us in determining which stat tools is suitable
    '''
    import numpy as np
    import os
    df_ls = []
    for fpath in tqdm ( xpath ):
        sbj = findall ( r's\d{2}', fpath, flags=re.IGNORECASE ) [0].upper ()
        sub_session = findall ( r'\d{6}', fpath ) [0]
        df = pd.read_pickle ( fpath )
        sess_N=fpath.split(sub_session)[-1].split(fpath)[0].split(os.sep)[0]
        dcomb=sub_session+sess_N
        df ['sub_session'] = sess_N
        df ['sess_N'] = sub_session
        df['dcomb']=dcomb
        df['sbj_id']=sbj

        df_ls.append ( df )

    # df_ls=[(pd.read_feather ( fpath )) for fpath in tqdm ( xpath )]
    df = pd.concat ( df_ls ).reset_index ( drop=True )
    # df.to_feather ( 'df_dev.feather')
    # df=pd.read_feather ( 'all_df.feather')

    df = df [['sbj_id', ncluster_col, 'type_exp', 'sub_session','sess_N','dcomb']]
    df ['nval'] = 1
    # df_opt = df.groupby ( ['sbj_id', 'type_exp', 'sub_session', ncluster_col] ).sum ()

    df_opt = pd.pivot_table(df, values='nval', index=['sbj_id', 'sub_session','sess_N','type_exp','dcomb'],
                    columns=[ncluster_col], aggfunc=np.sum,margins=True)

    if psave is not None:

        from old_code.gtheory.utils.misc import check_make_folder
        check_make_folder ( psave )
        psaves= f"{psave}all_subject_cluster_distribution_{ncluster_col}.xlsx"
        df_opt.to_excel ( psaves)

    return df_opt
