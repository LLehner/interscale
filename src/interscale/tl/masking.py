import torch
from torch_geometric.data import Batch

MASK_VALUE = 0


def apply_mask(batched_data: Batch):
    """Mask nodes from PyG object in .mask attribute.

    Args:
        batched_data (Batch): _description_

    Returns
    -------
        batched_data_w_mask (Batch):
            Batch only containing nodes that were not masked
        mask_idx (torch.Tensor):
            Indices of masked nodes

    Example:
        Data object:
        x = torch.tensor([[1., 2.], [3., 4.], [5., 6.], [7., 8.]])
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
        mask = torch.tensor([1, 0, 1, 0], dtype=torch.bool)
        data = Data(x=x, edge_index=edge_index, mask=mask)
        ----
        mask_idx = torch.tensor([1, 3])
        masked_values = torch.tensor([[0., 0.], [3., 4.], [0., 0.], [7., 8.]])
    """
    assert batched_data.mask is not None, "Mask is not set in the batch."

    mask = batched_data.mask
    mask_idx = torch.where(mask == 1)[0]  # TODO into 2D array [B, N_batched_nodes]
    masked_values = batched_data.x.clone()
    masked_values[mask] = MASK_VALUE
    batched_data_w_mask = batched_data.clone()
    batched_data_w_mask.x = masked_values
    return batched_data_w_mask, mask_idx


def create_transformer_attention_mask_from_edges(
    edge_index: torch.Tensor, batch: torch.Tensor, index_nodes: list, num_heads: int,
) -> torch.Tensor:
    """
    Long-range attention mask.

    Attention is allowed between all node pairs except:

    - local graph neighbors
    - self-attention

    CLS token attends to all nodes and all nodes attend to CLS.

    Returns
    -------
    torch.Tensor
        Shape:
            [num_heads * batch_size,
             max_seq_len + 1,
             max_seq_len + 1]

        Values:
            0      -> unmasked
            -inf   -> masked
    """
    INVALID_MASK_VALUE = float("-inf")

    batch_size = int(batch[-1].item() + 1)
    max_seq_len = max(len(nodes) for nodes in index_nodes)

    attention_mask = torch.zeros(
        (batch_size * num_heads,
         max_seq_len + 1,
         max_seq_len + 1),
        device=edge_index.device,
        dtype=torch.float32,
    )

    edge_set = {
        (int(src), int(dst))
        for src, dst in edge_index.t().tolist()
    }

    for b, nodes in enumerate(index_nodes):

        seq_len = len(nodes)

        batch_mask = torch.zeros(
            (seq_len + 1, seq_len + 1),
            device=edge_index.device,
            dtype=torch.float32,
        )

        node_to_local = {
            int(node): i
            for i, node in enumerate(nodes)
        }

        # mask local graph edges
        for src, dst in edge_set:

            if src not in node_to_local:
                continue

            if dst not in node_to_local:
                continue

            i = node_to_local[src]
            j = node_to_local[dst]

            batch_mask[i, j] = INVALID_MASK_VALUE
            batch_mask[j, i] = INVALID_MASK_VALUE

        # mask self attention
        diag_idx = torch.arange(seq_len, device=edge_index.device)

        batch_mask[diag_idx, diag_idx] = INVALID_MASK_VALUE

        # CLS token is last index and remains unmasked

        attention_mask[
            b * num_heads : (b + 1) * num_heads,
            : seq_len + 1,
            : seq_len + 1,
        ] = batch_mask

    return attention_mask


def attn_mask_diagonal(
    batch: torch.Tensor, index_nodes: list, num_heads: int, device: torch.device,
) -> torch.Tensor:
    """
    Mask self-attention between graph nodes.
    CLS remains unmasked.
    """
    batch_size = int(batch[-1].item() + 1)
    max_seq_len = max(len(nodes) for nodes in index_nodes)

    attention_mask = torch.zeros(
        (num_heads * batch_size,
         max_seq_len + 1,
         max_seq_len + 1),
        device=device,
        dtype=torch.float32,
    )

    diag_idx = torch.arange(max_seq_len, device=device)

    attention_mask[:, diag_idx, diag_idx] = float("-inf")

    return attention_mask
