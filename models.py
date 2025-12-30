import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv

class ScaledConv(nn.Module):
    """
    Wraps a GNN layer and scales its output by a learnable factor.
    """
    def __init__(self, conv, alpha_param):
        super().__init__()
        self.conv = conv
        self.alpha_param = alpha_param # Shared Parameter

    def forward(self, x, edge_index, **kwargs):
        out = self.conv(x, edge_index, **kwargs)
        trust_score = torch.sigmoid(self.alpha_param)
        # Normalize to prevent explosion in DistMult
        # Use eps to avoid division by zero if output is 0
        return F.normalize(out * trust_score, p=2, dim=-1, eps=1e-6)

class WeightedHeteroGraphConv(nn.Module):
    """
    A specific HeteroGraphConv that learns to trust different edge types differently.
    """
    def __init__(self, metadata, hidden_channels, out_channels, num_layers=1):
        super().__init__()
        self.metadata = metadata
        node_types, edge_types = metadata
        
        # 1. Trust Gate: Learnable scalar for each edge type
        self.edge_trust_weights = nn.ParameterDict()
        for edge_type in edge_types:
             key = f"{edge_type[0]}__{edge_type[1]}__{edge_type[2]}"
             self.edge_trust_weights[key] = nn.Parameter(torch.tensor(0.0))

        # 2. Structure: HeteroConv wrapping ScaledConv(SAGEConv)
        convs = {}
        for edge_type in edge_types:
            key = f"{edge_type[0]}__{edge_type[1]}__{edge_type[2]}"
            # Create base conv
            base_conv = SAGEConv((-1, -1), out_channels)
            # Wrap with the specific alpha parameter
            convs[edge_type] = ScaledConv(base_conv, self.edge_trust_weights[key])
            
        # We allow standard 'sum' aggregation now because scaling happens INSIDE
        self.hetero_conv = HeteroConv(convs, aggr='sum')

    def forward(self, x_dict, edge_index_dict):
        """
        Forward pass logic:
        HeteroConv calls ScaledConv for each edge.
        ScaledConv applies sigmoid(alpha).
        HeteroConv sums the results.
        """
        return self.hetero_conv(x_dict, edge_index_dict)

class DistMultDecoder(nn.Module):
    """
    DistMult Decoder for Link Prediction on Heterogeneous Graphs.
    Score(s, r, o) = <e_s, R_r, e_o> (element-wise product sum).
    """
    def __init__(self, metadata, hidden_channels):
        super().__init__()
        self.metadata = metadata
        node_types, edge_types = metadata
        
        # Relation embeddings: Diagonal matrix R_r for each relation type
        # Represented as a vector of size hidden_channels.
        # We use a ParameterDict keyed by the string representation of the edge type.
        self.rel_embeddings = nn.ParameterDict()
        for edge_type in edge_types:
            key = f"{edge_type[0]}__{edge_type[1]}__{edge_type[2]}"
            self.rel_embeddings[key] = nn.Parameter(torch.randn(hidden_channels))

    def forward(self, z_dict, edge_index_dict):
        """
        Computes scores for edges in edge_index_dict using embeddings in z_dict.
        Returns a dictionary of scores keyed by edge type.
        """
        scores_dict = {}
        for edge_type, edge_index in edge_index_dict.items():
            src_type, rel, dst_type = edge_type
            
            # Construct key to retrieve relation embedding
            key = f"{src_type}__{rel}__{dst_type}"
            if key not in self.rel_embeddings:
                continue
                
            r_emb = self.rel_embeddings[key]
            # Normalize relation embedding
            r_emb = F.normalize(r_emb, p=2, dim=-1)
            
            # Get node embeddings
            # We assume z_dict contains embeddings for the source and target node types
            if src_type not in z_dict or dst_type not in z_dict:
                continue

            x_src = z_dict[src_type]
            x_dst = z_dict[dst_type]
            
            # Select source and target embeddings based on edge indices
            # edge_index is [2, num_edges]
            src_indices = edge_index[0]
            dst_indices = edge_index[1]
            
            src_emb = x_src[src_indices] # [num_edges, hidden_channels]
            dst_emb = x_dst[dst_indices] # [num_edges, hidden_channels]
            
            # DistMult Score: sum(src * rel * dst, dim=1)
            # Element-wise product of source, relation, and target vectors, then sum
            score = (src_emb * r_emb * dst_emb).sum(dim=1)
            
            scores_dict[edge_type] = score
            
        return scores_dict
