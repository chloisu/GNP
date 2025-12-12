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
        # with the current setup, where the matrix is not normalized
        # at the beginning to test the normalize=True option in 
        # ResGConv of PyGGCN below, this is actually not redundant.
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
        # skip normalization step here, to check usage of 
        # normalize=True in ResGConv 
        self.AA = A.to(dtype)#scale_A_by_spectral_radius(A).to(dtype)
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
            self.gconv.append(ResGConv(embed,normalize=True))
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

