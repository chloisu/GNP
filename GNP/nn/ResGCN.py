import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from GNP.utils import scale_A_by_spectral_radius 
from torch_geometric.nn.models import MLP as PyGMLP
from torch_geometric.utils import is_torch_sparse_tensor,dense_to_sparse, to_edge_index, to_torch_sparse_tensor
from torch_geometric.data import Data, Batch
from torch_geometric.contrib.nn import ResGConv as ResGConv
# if published, import from package, else use below


"""
# ResGConv imports
from typing import Optional
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils import add_self_loops as add_self_loops_fn
from torch_geometric.utils import (
    is_torch_sparse_tensor,
    scatter,
    spmm,
    to_edge_index,
)
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils.sparse import set_sparse_value
"""

#-----------------------------------------------------------------------------
# An MLP layer.
class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, num_layers, hidden, drop_rate,
                 use_batchnorm=False, is_output_layer=False):
        super().__init__()
        self.num_layers = num_layers
        self.use_batchnorm = use_batchnorm
        self.is_output_layer = is_output_layer

        self.lin = nn.ModuleList()
        self.lin.append( nn.Linear(in_dim, hidden) )
        for i in range(1, num_layers-1):
            self.lin.append( nn.Linear(hidden, hidden) )
        self.lin.append( nn.Linear(hidden, out_dim) )
        if use_batchnorm:
            self.batchnorm = nn.ModuleList()
            for i in range(0, num_layers-1):
                self.batchnorm.append( nn.BatchNorm1d(hidden) )
            if not is_output_layer:
                self.batchnorm.append( nn.BatchNorm1d(out_dim) )
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, R):                              # R: (*, in_dim)
        assert len(R.shape) >= 2
        for i in range(self.num_layers):
            R = self.lin[i](R)                            # (*, hidden)
            if i != self.num_layers-1 or not self.is_output_layer:
                if self.use_batchnorm:
                    shape = R.shape
                    R = R.view(-1, shape[-1])
                    R = self.batchnorm[i](R)
                    R = R.view(shape)
                R = self.dropout(F.relu(R))
                                                          # (*, out_dim)
        return R
    

#-----------------------------------------------------------------------------
# A GCN layer.
class GCNConv(nn.Module):

    def __init__(self, AA, in_dim, out_dim):
        super().__init__()
        self.AA = AA  # normalized A
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, R):                         # R: (n, batch_size, in_dim)
        assert len(R.shape) == 3
        n, batch_size, in_dim = R.shape
        assert in_dim == self.in_dim
        if in_dim > self.out_dim:
            R = self.fc(R)                           # (n, batch_size, out_dim)
            R = R.view(n, batch_size * self.out_dim) # (n, batch_size * out_dim)
            R = self.AA @ R                          # (n, batch_size * out_dim)
            R = R.view(n, batch_size, self.out_dim)  # (n, batch_size, out_dim)
        else:
            R = R.view(n, batch_size * in_dim)       # (n, batch_size * in_dim)
            R = self.AA @ R                          # (n, batch_size * in_dim)
            R = R.view(n, batch_size, in_dim)        # (n, batch_size, in_dim)
            R = self.fc(R)                           # (n, batch_size, out_dim)
        return R


#-----------------------------------------------------------------------------
# GCN with residual connections.
class ResGCN(nn.Module):
    
    def __init__(self, A, num_layers, embed, hidden, drop_rate,
                 scale_input=True, dtype=torch.float32):
        # A: float64, already on device.
        #
        # For graph convolution, A will be normalized and cast to
        # lower precision and named AA.
        
        super().__init__()
        self.dtype = dtype # used by GNP.precond.GNP
        self.num_layers = num_layers
        self.embed = embed
        self.scale_input = scale_input

        # Note: scale_A_by_spectral_radius() has been called when
        # defining the problem; hence, it is redundant. We keep the
        # code here to leave open the possibility of normalizing A in
        # another manner.
        self.AA = scale_A_by_spectral_radius(A).to(dtype)

        self.mlp_initial = MLP(1, embed, 4, hidden, drop_rate)
        self.mlp_final = MLP(embed, 1, 4, hidden, drop_rate,
                             is_output_layer=True)
        self.gconv = nn.ModuleList()
        self.skip = nn.ModuleList()
        self.batchnorm = nn.ModuleList()
        for i in range(num_layers):
            self.gconv.append( GCNConv(self.AA, embed, embed) )
            self.skip.append( nn.Linear(embed, embed) )
            self.batchnorm.append( nn.BatchNorm1d(embed) )
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, r):                        # r: (n, batch_size)
        assert len(r.shape) == 2
        n, batch_size = r.shape
        if self.scale_input:
            scaling = torch.linalg.vector_norm(r, dim=0) / np.sqrt(n)
            r = r / scaling  # scaling
        r = r.view(n, batch_size, 1)                # (n, batch_size, 1)
        R = self.mlp_initial(r)                     # (n, batch_size, embed)
        
        for i in range(self.num_layers):
            R = self.gconv[i](R) + self.skip[i](R)  # (n, batch_size, embed)
            R = R.view(n * batch_size, self.embed)  # (n * batch_size, embed)
            R = self.batchnorm[i](R)                # (n * batch_size, embed)
            R = R.view(n, batch_size, self.embed)   # (n, batch_size, embed)
            R = self.dropout(F.relu(R))             # (n, batch_size, embed)
            
        z = self.mlp_final(R)                       # (n, batch_size, 1)
        z = z.view(n, batch_size)                   # (n, batch_size)
        if self.scale_input:
            z = z * scaling  # scaling back
        return z


class PyGGCN(nn.Module):
    """
    A PyTorch Geometric implementation of the GNP ResGCN architecture.

    This module wraps the original ResGCN design so that graph convolution
    is performed using PyG’s `ResGConv` operator, while preserving the
    scale-equivariance behavior required by GNP.

    Inputs
    ------
    A : torch.Tensor
        Graph adjacency matrix (dense or sparse), already on device.
    num_layers : int
        Number of ResGConv layers.
    embed : int
        Node embedding dimension.
    hidden : int
        Hidden dimension for the MLPs.
    drop_rate : float
        Dropout probability.
    scale_input : bool
        Whether to apply global input/output normalization.
    dtype : torch.dtype
        Internal dtype used for computations.
    """

    def __init__(self, A, num_layers, embed, hidden, drop_rate,
                 scale_input=True, dtype=torch.float32):

        super().__init__()
        self.dtype = dtype
        self.num_layers = num_layers
        self.embed = embed
        self.scale_input = scale_input

        # Normalize A by its spectral radius.
        # Retain the call here for modularity.
        self.AA = scale_A_by_spectral_radius(A).to(dtype)

        # Convert adjacency to PyG’s (edge_index, edge_weight) format.
        if is_torch_sparse_tensor(self.AA):
            edge_index, edge_weight = to_edge_index(self.AA)
        else:
            edge_index, edge_weight = dense_to_sparse(self.AA)

        # Register adjacency components as buffers so they migrate with `.to(device)`.
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_weight", edge_weight)

        # Input MLP: maps scalar RHS entries → embedding dimension.
        self.mlp_initial = PyGMLP(
            in_channels=1,
            hidden_channels=hidden,
            out_channels=embed,
            num_layers=4,
            dropout=drop_rate,
            norm=None,
        )

        # Output MLP: maps embeddings → scalar prediction.
        self.mlp_final = PyGMLP(
            in_channels=embed,
            hidden_channels=hidden,
            out_channels=1,
            num_layers=4,
            dropout=[drop_rate] * (4 - 1) + [0.0],  # No dropout in final layer.
            norm=None,
        )

        # ResGConv layers with batch normalization.
        self.gconv = nn.ModuleList()
        self.batchnorm = nn.ModuleList()
        for _ in range(num_layers):
            self.gconv.append(ResGConv(embed))
            self.batchnorm.append(nn.BatchNorm1d(embed))

        self.dropout = nn.Dropout(drop_rate)

    def forward(self, r):
        """
        Parameters
        ----------
        r : torch.Tensor of shape (n, batch_size)
            Right-hand-side vectors stacked column-wise.

        Returns
        -------
        torch.Tensor of shape (n, batch_size)
            Network predictions corresponding to each RHS.
        """
        assert r.dim() == 2
        n, batch_size = r.shape

        # Optional normalization for scale-equivariance.
        if self.scale_input:
            scaling = torch.linalg.vector_norm(r, dim=0) / np.sqrt(n)
            r = r / scaling

        r = r.to(self.dtype)
        edge_index = self.edge_index.to(r.device)
        edge_weight = (
            self.edge_weight.to(r.device)
            if self.edge_weight is not None
            else None
        )

        # Construct PyG Data objects, one per RHS.
        data_list = []
        for i in range(batch_size):
            x_i = r[:, i].view(-1, 1)
            data_list.append(Data(x=x_i, edge_index=edge_index, edge_attr=edge_weight))

        # Batch all graphs for vectorized processing.
        batch = Batch.from_data_list(data_list)
        x = batch.x
        edge_index = batch.edge_index
        edge_weight = batch.edge_attr

        # Input MLP.
        x = self.mlp_initial(x)

        # GNN layers.
        for i in range(self.num_layers):
            x = self.gconv[i](x, edge_index, edge_weight)
            x = self.batchnorm[i](x)
            x = self.dropout(F.relu(x))

        # Output MLP.
        x = self.mlp_final(x)

        # Unbatch into shape (n, batch_size).
        x = x.view(batch_size, n, 1)
        x = x.transpose(0, 1).squeeze(-1)

        if self.scale_input:
            x = x * scaling

        return x


"""
def res_gconv_norm(  # noqa: F811
    edge_index: Adj,
    edge_weight: OptTensor = None,
    num_nodes: Optional[int] = None,
    improved: bool = False,
    add_self_loops: bool = True,
    flow: str = "source_to_target",
    dtype: Optional[torch.dtype] = None,
):
    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        assert edge_index.size(0) == edge_index.size(1)

        adj_t = edge_index

        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = torch_sparse.fill_diag(adj_t, fill_value)

        # Implementation from https://github.com/jiechenjiechen/GNP
        # GNP/utils.py scale_A_by_spectral_radius(A)

        adj_t = adj_t.to_dense()
        absA = adj_t.abs()
        m, n = absA.shape
        row_sum = absA @ torch.ones(n, 1, dtype=absA.dtype, device=absA.device)
        col_sum = torch.ones(1, m, dtype=absA.dtype, device=absA.device) @ absA
        gamma = torch.min(torch.max(row_sum), torch.max(col_sum))

        adj_t = adj_t * (1. / gamma.item())
        adj_t = SparseTensor.from_dense(adj_t)
        return adj_t

    if is_torch_sparse_tensor(edge_index):
        assert edge_index.size(0) == edge_index.size(1)

        if edge_index.layout == torch.sparse_csc:
            raise NotImplementedError("Sparse CSC matrices are not yet "
                                      "supported in 'res_gconv_norm'")

        adj_t = edge_index
        if add_self_loops:
            adj_t, _ = add_self_loops_fn(adj_t, None, fill_value, num_nodes)

        edge_index, value = to_edge_index(adj_t)

        # Implementation from https://github.com/jiechenjiechen/GNP
        # GNP/utils.py scale_A_by_spectral_radius(A)

        absA = torch.absolute(adj_t)
        m, n = absA.shape
        row_sum = absA @ torch.ones(n, 1, dtype=adj_t.dtype,
                                    device=adj_t.device)
        col_sum = torch.ones(1, m, dtype=adj_t.dtype,
                             device=adj_t.device) @ absA
        gamma = torch.min(torch.max(row_sum), torch.max(col_sum))
        value = value / gamma

        return set_sparse_value(adj_t, value), None

    assert flow in ['source_to_target', 'target_to_source']
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if add_self_loops:
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    row, col = edge_index[0], edge_index[1]
    gamma_row = scatter(torch.abs(edge_weight), row, dim_size=num_nodes,
                        reduce='sum')
    gamma_col = scatter(torch.abs(edge_weight), col, dim_size=num_nodes,
                        reduce='sum')
    gamma = min(max(gamma_row), max(gamma_col))
    edge_weight = edge_weight / gamma

    return edge_index, edge_weight
"""

#
#class ResGConv(MessagePassing):
#    r"""The graph convolutional operator with residual skip connections from
#    the `"Graph Neural Preconditioners for Iterative Solutions of Sparse
#    Linear Systems" <https://arxiv.org/pdf/2406.00809>`_ paper.
#
#    .. math::
#        \text{Res-GCONV}(\mathbf{X}) = \text{ReLU}\left(\mathbf{XU} +
#        \mathbf{\hat{A}XW}\right)
#
#    where
#    :math:`\mathbf{\hat{A}}\in \mathbb{R}^{n\times n}` denotes the weighted
#    adjacency matrix
#    normalized using the normalization
#    :math:`\mathbf{\hat{A}} = \mathbf{A}/\gamma`, where :math:`\gamma =
#    \min\{\max_j\{\sum_j
#    |a|_{ij}\},\max_i\{\sum_i |a|_{ij}\}\}`
#    and :math:`\mathbf{W,U}` are matrices containing the learnable parameters.
#
#    Args:
#        channels (int): Size of each input and output sample.
#            bias_u (bool, optional): If set to :obj:`False`, the layer will not
#            learn an additive bias together with :math:`\mathbf{U}`.
#            (default: obj:`True`)
#        bias_u (bool, optional): If set to :obj:`False`, the layer will not
#            learn an additive bias together with :math:`\mathbf{U}`.
#            (default: obj:`True`)
#        add_self_loops (bool, optional): If set to :obj:`True`, will add
#            self-loops to the input graph. (default: :obj:`False`)
#        cached (bool, optional): If set to :obj:`True`, the layer will cache
#            the computation of :math:`\mathbf{\hat{A}} = \mathbf{A}/\gamma` on
#            first execution, and will use the cached version for further
#            executions. This parameter should only be set to :obj:`True` in
#            transductive learning scenarios. (default: :obj:`False`)
#        normalize (bool, optional): Whether to apply normalization
#            by :math:`\gamma`. (default: :obj:`True`)
#        **kwargs (optional): Additional arguments of
#            :class:`torch_geometric.nn.conv.MessagePassing`.
#
#    Shapes:
#        - **input:**
#          node features :math:`(|\mathcal{V}|, F)`,
#          edge indices :math:`(2, |\mathcal{E}|)`,
#          edge weights :math:`(|\mathcal{E}|)` *(optional)*
#        - **output:** node features :math:`(|\mathcal{V}|, F)`
#    """
#
#    _cached_edge_index: Optional[OptPairTensor]
#    _cached_adj_t: Optional[SparseTensor]
#
#    def __init__(self, channels: int, bias_u: bool = True, bias_w: bool = True,
#                 add_self_loops: bool = False, cached: bool = False,
#                 normalize: bool = True, **kwargs):
#
#        kwargs.setdefault('aggr', 'add')
#        super().__init__(**kwargs)
#
#        self.channels = channels
#        self.cached = cached
#        self.normalize = normalize
#        self.add_self_loops = add_self_loops
#        self.bias_u = bias_u
#        self.bias_w = bias_w
#
#        self._cached_edge_index = None
#        self._cached_adj_t = None
#
#        self.U = Linear(self.channels, self.channels, bias=self.bias_u)
#        self.W = Linear(self.channels, self.channels, bias=self.bias_w)
#
#        self.reset_parameters()
#
#    def reset_parameters(self):
#        super().reset_parameters()
#        self.U.reset_parameters()
#        self.W.reset_parameters()
#        self._cached_edge_index = None
#        self._cached_adj_t = None
#
#    def forward(self, x: Tensor, edge_index: Adj,
#                edge_weight: OptTensor = None) -> Tensor:
#
#        if self.normalize:
#            if isinstance(edge_index, Tensor):
#                cache = self._cached_edge_index
#                if cache is None:
#                    edge_index, edge_weight = res_gconv_norm(  # yapf: disable
#                        edge_index, edge_weight, x.size(self.node_dim), False,
#                        self.add_self_loops, self.flow, dtype=x.dtype)
#                    if self.cached:
#                        self._cached_edge_index = (edge_index, edge_weight)
#                else:
#                    edge_index, edge_weight = cache[0], cache[1]
#
#            elif isinstance(edge_index, SparseTensor):
#                cache = self._cached_adj_t
#                if cache is None:
#                    edge_index = res_gconv_norm(  # yapf: disable
#                        edge_index, edge_weight, x.size(self.node_dim), False,
#                        self.add_self_loops, self.flow, dtype=x.dtype)
#                    if self.cached:
#                        self._cached_adj_t = edge_index
#                else:
#                    edge_index = cache
#
#        wx = self.W(x)
#        awx = self.propagate(edge_index, x=wx, edge_weight=edge_weight)
#        out = awx
#        out = out + self.U(x)
#
#        return out
#
#    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
#        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
#
#    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
#        return spmm(adj_t, x, reduce=self.aggr)
#
#    def __repr__(self) -> str:
#        return (f'{self.__class__.__name__}({self.channels}, '
#                f'bias_u={self.bias_u}, bias_w={self.bias_w})')
#