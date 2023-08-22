import leidenalg as la
import igraph as ig
import numpy as np
import pandas as pd
'''



1) coupling strengths (ω) and resolution parameter (γ) are labeled/known as interslice_weight and resolution_parameter

Yes, this is correct. The interslice_weight in time_slices_to_layers is identical to ω, and the resolution_parameter 
for the various quality functions is identical to γ. In the original paper by Mucha et al., they use modularity, 
corresponding to the resolution parameter for RBConfigurationVertexPartition. Of course, 
for other quality functions, e.g. CPMVertexPartition, you don’t get the exact same formulation, 
but the idea remains the same.


2) Is it common practice, at least in leidenalg to set two different resolution_parameter when 
determining the partition for interslice and intraslice?

Yes, this is absolutely necessary. If you look carefully at the formulation, the interslice couplings are 
always positive. That is, there should be no cost in putting clusters in separate slices in the same 
community. For that reason, you need to set the resolution_parameter for any interslice couplings to 0, 
otherwise there will be some costs incurred.
'''


# https://leidenalg.readthedocs.io/en/stable/reference.html#leidenalg.find_partition_multiplex
def make_slice_name (arr_temporal):
    nslice = arr_temporal.shape [0]
    slice_name = [f'Slice {idx}' for idx in range ( nslice )]
    return slice_name


class gt_leidenalg:
    def __init__ (self, arr_temporal, ch_names, slice_name=None, drop_ch=None,
                  n_iterations=3, interslice_weight=1, resolution_parameter=1,community_output=None):
        '''
        community_output: can be either dataframe or numpy_arr
        Parameters
        ----------
        arr_temporal
        ch_names
        slice_name
        drop_ch
        n_iterations
        interslice_weight
        resolution_parameter
        community_output
        '''
        self.arr_con = arr_temporal
        # if ch_names is None:
        #     B=1
        # else:
        #     self.ch_names = ch_names
        self.ch_names = ch_names
        self.slice_name = make_slice_name ( arr_temporal ) if slice_name is None else slice_name
        self.community_output='dataframe' if community_output is None else community_output

        if drop_ch:
            self.drop_ch = drop_ch
            self.all_idx=self.drop_vertex()
        else:
            self.all_idx=None

        self.G_list = [self.arr_to_graph ( self.arr_con [my_con_idx, :, :], self.ch_names )
                       for my_con_idx, ep_name in enumerate ( self.slice_name )]
        self.n_iterations = n_iterations
        self.interslice_weight = interslice_weight
        self.resolution_parameter = resolution_parameter
        self.G_layers, self.G_interslice, self.G = [], [], []
        self.partition_all = []
        self.partitions = []
        self.diff = []
        self.interslice_partition = []

    def time_slices_to_layers (self):
        # We loop slice by slice
        # Note that, we need not to reshape the arr_temporal

        # # Note here, the slice', vertex_id_attr is assigned to `id` since we define it as id at the previous step
        self.G_layers, self.G_interslice, self.G = la.time_slices_to_layers ( self.G_list,
                                                                              interslice_weight=self.interslice_weight,
                                                                              slice_attr='slice',
                                                                              vertex_id_attr='id',
                                                                              edge_type_attr='type',
                                                                              weight_attr='weight' )


    def arr_to_graph (self,arr, label):
        # Convert from slices to layers
        G = ig.Graph.Weighted_Adjacency ( arr.tolist () )
        G.vs ['id'] = label
        # drop_label=[]
        # G.delete_vertices([1,2,3,4,5,6,7,8,9])
        if self.all_idx is not None:
            G.delete_vertices ( self.all_idx )
        return G

    def drop_vertex (self):

        all_idx = []
        for idx, ch in enumerate ( self.ch_names ):
            if ch in self.drop_ch:
                all_idx.append ( idx )
        return all_idx

    def create_partitions (self):
        # # Create partitions

        self.partitions = [la.CPMVertexPartition ( H, node_sizes='node_size', weights='weight',
                                                   resolution_parameter=self.resolution_parameter ) for H in
                           self.G_layers]
        self.interslice_partition = la.CPMVertexPartition ( self.G_interslice, resolution_parameter=0,
                                                            node_sizes='node_size', weights='weight' )

        # # Detect communities
        optimiser = la.Optimiser ()
        

        gg=np.random.randint(100000)
        optimiser.set_rng_seed (gg)
        self.diff = optimiser.optimise_partition_multiplex ( self.partitions + [self.interslice_partition],
                                                             n_iterations=self.n_iterations )

        # Plot network
        # Note here, the`v` is assigned to `id` since we define it as id at the previous step
        self.partition_all = ig.VertexClustering ( self.G, self.partitions [0].membership )

    def get_membership (self,quality_val=False):

        membership_slice = {}
        for v, m in zip ( self.G.vs, self.partitions [0].membership ):
            if v ['slice'] not in membership_slice:
                membership_slice [v ['slice']] = {}
            membership_slice [v ['slice']] [v ['id']] = m
        # a = np.array(membership_slice)
        membership_d = pd.DataFrame ( membership_slice)
        membership_d.set_axis(self.slice_name, axis=1, inplace=True)
        # self.slice_name
        # membership_dxxx  = membership_d .values
        if self.community_output=='numpy_arr':
            membership_d  = membership_d .values

        if quality_val:
            return membership_d,self.diff
            
        return membership_d

    def arr_to_membership (self, n_iterations=None, interslice_weight=None, resolution_parameter=None,
                           community_output=None,quality_val=False):
        '''

        Extract only the membership
        coupling strengths (ω:omega) and resolution parameter (γ:gamma) are labeled/known as
        interslice_weight and resolution_parameter

        Returns
        -------
        gamma=0.3,n_iterations=2,
        interslice_weight = 1e-1
        '''
        self.n_iterations = self.n_iterations if n_iterations is None else n_iterations
        self.interslice_weight = self.interslice_weight if interslice_weight is None else interslice_weight
        self.resolution_parameter = self.resolution_parameter if resolution_parameter is None else resolution_parameter
        self.community_output=self.community_output if community_output is None else community_output

        self.time_slices_to_layers ()
        self.create_partitions ()
        return self.get_membership (quality_val=quality_val)

    @staticmethod
    def _viz_membership_temporal (G, partition_all):
        import igraph as ig
        ig.plot ( partition_all, bbox=(0, 0, 300, 300),
                  vertex_label=[f'{v ["id"]}-{v ["slice"]}' for v in G.vs] )

    def viz_membership_temporal (self):
        '''

        Calculate membership and show via viz
        Returns
        -------

        '''
        self.time_slices_to_layers ()
        
        self.create_partitions ()
        # Temporary disable
        # self._viz_membership_temporal ( self.G, self.partition_all )
