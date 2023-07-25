import os
import pickle
from os.path import (join)

import pandas as pd
from sklearn.cluster import KMeans


def viz_prediction_kmeans (data, kmeans):
    import matplotlib.pyplot as plt
    # https://stackoverflow.com/a/14762601/6446053
    y_kmeans = kmeans.predict ( data )
    fig, ax1 = plt.subplots ()
    ax1.scatter ( data [:, 0], data [:, 1], c=y_kmeans, s=50,
                  cmap='viridis' )

    #
    centers = kmeans.cluster_centers_
    ax1.scatter ( centers [:, 0], centers [:, 1], c='red', s=600, alpha=0.5 );

    # ax2 = ax1.twinx()
    #
    # # horizontal line
    ax1.hlines ( y=0.7, xmin=0, xmax=0.7, color='red', zorder=1 )
    # # vertical line
    ax1.vlines ( x=0.7, ymin=0, ymax=0.7, color='red', zorder=2 )

    ax1.axvline ( x=2.1, color='r', label='axvline - full height' )
    #
    ax1.axhline ( y=2.1, color='r', label='axvline - full height' )
    [ax1.text ( x, y, texts, c='red', fontsize=20 ) for x, y, texts in
     zip ( [0.2, 1.7, 1.7, 2.2, 2.2], [0.2, 1.4, 2.2, 1.4, 2.2], ['A', 'B', 'E', 'C', 'D'] )]
    ax1.set_xlabel ( 'Local RT (s)' )
    ax1.set_ylabel ( 'Global RT (s)' )
    ax1.set_title ( 'Rt labeling using kmeans' )
    constant_converion = 5000 / 3600

    def a2b (y):
        return y * constant_converion

    def b2a (y):
        return y / constant_converion

    secax = ax1.secondary_yaxis ( 'right', functions=(a2b, b2a) )
    secax.set_yticks ( [a2b ( y ) for y in ax1.get_yticks ()] )
    secax.set_ylabel ( 'Deviation Distance (m)' )
    # plt.show()
    b = 1


def fit_kmeans (df_list, n_clusters, **kwargs):
    psave= kwargs.get('psave',None)
    session= kwargs.get('session',None)
    if psave is not None and session is None :
        fname=f'kmean_cluster_{n_clusters}.pickle'
    else:
        fname=f'kmean_cluster_{n_clusters}_ss_{session}.pickle'

    spath=os.path.join(psave,fname)
    df = pd.concat ( df_list ).reset_index ( drop=True )
    data = df [['rt_local_sec', 'rt_global_sec']].dropna ().to_numpy ()
    kmeans = KMeans ( init='k-means++', n_clusters=n_clusters, n_init=10 ).fit ( data )
    
    
    
    # It is important to use binary access
    if psave is not None:
        from gtheory.utils.misc import check_make_folder
        check_make_folder ( psave )
        with open ( spath, 'wb' ) as f:
            pickle.dump ( kmeans, f )
    return kmeans


class predict_kmeans_model:
    def __init__ (self, fpath):
        import re
        self.ncluster=re.search(r"(?<=kmean_cluster_).*?(?=_)", fpath).group(0)
        # j=re.search ( r'kmean_(.*?).pickle', fpath ).group ( 1 )
        # re.findall(r's\d+|\d+m', j)
        # self.ncluster = re.search ( r'kmean_cluster_(.*?).pickle', fpath ).group ( 1 )
        # Load the model
        with open ( fpath, 'rb' ) as f:
            self.kmeans = pickle.load ( f )

    def predict_cluster (self, df, fnextension=None,psave=None,sbj_id=None):
        '''
        Define class identity for each events (i.e.,each deviation onset) based on the kmeans model. The result will
        be logged and saved into new df file with the name `filtered-event.feather`
        '''
        
        try:
            # df = df.dropna ().reset_index ( drop=True )
            col_name = f'ncluster_{self.ncluster}'
            try:
                df [col_name] = self.kmeans.predict (
                    df [['rt_local_sec', 'rt_global_sec']].to_numpy () )
                if psave is not None:
                    new_path = join ( psave, fnextension )
                    df.to_feather ( new_path )
                return df
            except ValueError:
                print(f'Issue with subject {sbj_id}. ValueError')
                return df
        except AttributeError:
            print(f'Issue with subject {sbj_id}. AttributeError')
            return df
            
