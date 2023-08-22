import warnings

import numpy as np
import scipy.stats as stats
import seaborn as sns
from matplotlib import pyplot as plt


def _jointplot_scatter(df_flabel,nrt,nvar):
    g=sns.jointplot(data=df_flabel, y=nrt, x=nvar,kind = 'reg')
    a=df_flabel[nrt]
    b=df_flabel[nvar]
    
    try:
        r, p = stats.pearsonr(a, b)
    except ValueError:
        warnings.warn('Most probably single points,This only warning')
        r, p=np.NAN,np.NAN
    g.ax_joint.annotate(f'$\\rho = {r:.3f}, p = {p:.3f}$',xy=(0.1, 0.9), xycoords='axes fraction',
        ha='left', va='center',
        bbox={'boxstyle': 'round', 'fc': 'powderblue', 'ec': 'navy'})
    plt.suptitle(f'{nrt}_{nvar}')
    
    return g
def scatterplot_correlation_comm_measure(df,dis_analysis,dis_cluster,measure=''):
    
    roi_unique=['frontal','temporalleft','temporalright','central','occipital']
    rt_list=['rt_local','rt_global']
    # roi_unique=['frontal']
    # rt_list=['rt_local']
    # roi_unique=df['brain_region'].unique().tolist()
    nlabel_unique=df[dis_cluster].unique().tolist()
    # fig, axs = plt.subplots(len(rt_list),len(nvars))
    # https://stackoverflow.com/a/64589628/6446053
    # shorturl.at/dluN1
    all_fig=[]
    all_subnote=[]
    for dislabel in nlabel_unique:
        df_flabel=df[df[dis_cluster]==dislabel].reset_index(drop=True)
        
        for idx_rt, nrt in enumerate(rt_list):
            plt.close()
            for idx_var, nvar  in enumerate(roi_unique):
                g=_jointplot_scatter(df_flabel,nrt,nvar)
                # plt.show()
                # title=f"{measure} cor between {nrt}:{nvar}  \n for Fatigue Condi:{dislabel}"
                # plt.suptitle(title)
                # plt.show()
                # k=1
                title_long=f"{dis_analysis}\n cor between {nrt}:{nvar} {measure} \n for Fatigue Condi:{dislabel}"
                all_subnote.append(title_long)
                all_fig.append(g.fig)
                # plt.clf()
                # plt.show()
    
    return all_fig,all_subnote

def scatterplot_correlation_ncommunities(df,nvars):
  h=1
  rt_list=['rt_local','rt_global']
  # fig, axs = plt.subplots(len(rt_list),len(nvars))
  # https://stackoverflow.com/a/64589628/6446053
  # shorturl.at/dluN1
  all_fig=[]
  for idx_rt, nrt in enumerate(rt_list):
    for idx_var, nvar  in enumerate(nvars):
        g=_jointplot_scatter(df,nrt,nvar)
        all_fig.append(g.fig)
        # plt.show()

  return all_fig

def viz_comm_measure_club(df,fsub_analysis,xvar=None,yvar=None,var_list=None) :
    
    sns.set_theme(style="whitegrid")
    
    nlabel=len(var_list)
    if nlabel!=1:
        fig, axs = plt.subplots(nrows=nlabel)
        
        for idx, dis_condi  in enumerate(var_list):
            df_filter=df[df.mental_condition==dis_condi]
            sns.boxplot(x=xvar, y=yvar, data=df_filter,ax=axs[idx]).set_title(f'Condition {dis_condi}')
    else:
        for idx, dis_condi  in enumerate(var_list):
            fig = plt.figure()
            df_filter=df[df.mental_condition==dis_condi]
            sns.boxplot(x=xvar, y=yvar, data=df_filter).set_title(f'Condition {dis_condi}')
            # j=2
    plt.legend([],[], frameon=False)
    plt.suptitle(fsub_analysis)
    plt.tight_layout()
    # plt.show()
    
    return fig,fsub_analysis


def get_color (nwindows):
    # https://stackoverflow.com/a/14720445/6446053
    # all_colors = [k for k, v in pltc.cnames.items ()]
    all_colors = [
        'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
        'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
        'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
    color_ls = all_colors [:nwindows]
    return color_ls

def viz_line_overlay():
    '''
    Similar to Braun2015 Figure 1 (F)
    Returns
    -------

    '''
    # import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    import matplotlib.colors as mcolors
    # nn=mcolors.CSS4_COLORS
    all_colors=list(mcolors.TABLEAU_COLORS.keys())
    
    b=1
    np.random.seed(0)
    rng = np.random.default_rng(2)
    mlist=[]
    for _ in range(4):
        
        m=np.random.rand(4).tolist()
        n=rng.integers(0, 6, size=(1)).tolist()*4
        df = pd.DataFrame(zip(m,n), columns=['yval','type'])
        mlist.append(df)
    
    df=pd.concat(mlist).reset_index(drop=True).reset_index()
    # df.to_feather('test.feather')
    # df=pd.read_feather('test.feather')
    df['C'] = df['type'].diff()
    df['C']=df['C'].fillna(10)
    
    nb=df.type[(df['C'] != 0)].to_frame().reset_index()
    unique_val=nb['type'].drop_duplicates().sort_values().tolist()
    ngroup_type=dict(zip(unique_val,[f'type {idx}' for idx in unique_val]))
    nb['ngroup']=nb["type"].map(ngroup_type)
    color_group=all_colors[:len(unique_val)]
    res = dict(zip(unique_val, color_group))
    nb["color"] = nb["type"].map(res)
    # y=df['yval'].values
    # x=df["index"].values
    # fig, ax = plt.subplots()
    
    starting_point=nb["index"].values.tolist()
    mcolor=nb["color"].values.tolist()
    group_type=nb["ngroup"].values.tolist()
    nspace=4
    nheight=1
    fg=sns.lineplot(data=df, x="index", y="yval")
    for ncolor,spoint,gtype in zip(mcolor,starting_point,group_type):
        fg.axes.add_patch(patches.Rectangle((spoint, 0),
            nspace,nheight,edgecolor = 'blue',
            facecolor = ncolor,fill=True,alpha=0.1,ls=':') )
        fg.axes.text(spoint+1.5, 0.1, gtype , size=10,
            va="baseline", ha="left", multialignment="left")
    plt.show()


def viz_isometric_heatmap (data, N, background_style=None):
    color_list = get_color ( data.shape [0] )
    
    if background_style:
        plt.style.use ( 'dark_background' )
    
    ax = plt.figure ().add_subplot ( projection='3d' )
    for i, (plane, cmap) in enumerate ( zip ( data, color_list ) ):
        indices = np.indices ( (N, N) )
        norm = plt.Normalize ( plane.min (), plane.max () )
        ax.bar ( left=indices [0].ravel (), bottom=indices [1].ravel (), height=0.9,
            zs=i, zdir='y',
            color=plt.get_cmap ( cmap ) ( norm ( plane ).ravel () ) )


def viz_connectivity_circle (arr_temporal, ch_names, slice_name):
    from mne.viz import circular_layout, plot_connectivity_circle
    node_order = ch_names
    node_angles = circular_layout ( ch_names, node_order, start_pos=90,
        group_boundaries=[0, len ( ch_names ) // 2] )
    fig = plt.figure ( num=None, figsize=(5, 5), facecolor='black' )
    
    for my_con_idx, ep_name in enumerate ( slice_name ):
        my_con = arr_temporal [my_con_idx, :, :]
        plot_connectivity_circle ( my_con, ch_names, n_lines=300,
            node_angles=node_angles,
            title=f'Slice: {ep_name}', padding=0, fontsize_colorbar=6,
            fig=fig )
        plt.show ()
