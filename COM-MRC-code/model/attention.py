import torch
import torch.nn as nn
from torch.nn import functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self,num_heads,input_depth, total_key_depth,
                 total_value_depth, output_depth,dropout=0.0):
        super().__init__()
        assert total_key_depth % num_heads==0
        assert total_value_depth % num_heads==0

        self.num_heads = num_heads
        self.query_scale = (total_key_depth//num_heads) ** -0.5

        self.query_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.key_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.value_linear = nn.Linear(input_depth, total_value_depth, bias=False)
        self.output_linear = nn.Linear(total_value_depth, output_depth, bias=False)

        self.dropout = nn.Dropout(dropout)
    def _split_heads(self, x):
        """
        Split x such to add an extra num_heads dimension
        Input:
            x: a Tensor with shape [batch_size, seq_length, depth]
        Returns:
            A Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        """

        if len(x.shape) != 3:
            raise ValueError("x must have rank 3")
        shape = x.shape
        return x.view(shape[0], shape[1], self.num_heads, shape[2]//self.num_heads).permute(0, 2, 1, 3)
    def _merge_heads(self, x):
        """
        Merge the extra num_heads into the last dimension
        Input:
            x: a Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        Returns: x.permute(0, 2, 1, 3).reshape(shape[0],shape[2],shape[1]*shape[3])
            A Tensor with shape [batch_size, seq_length, depth]
        """
        if len(x.shape) != 4:
            raise ValueError("x must have rank 4")
        shape = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(shape[0], shape[2], shape[3]*self.num_heads)

    def forward(self, queries, keys, values):

        queries = self.query_linear(queries)
        queries = self._split_heads(queries) # [batch_size, num_heads, seq_length, depth/num_heads]

        keys = self.key_linear(keys)
        values = self.value_linear(values)
        keys = self._split_heads(keys)
        values = self._split_heads(values)

        # scale queries
        queries *= self.query_scale
        # logits = torch.einsum('ijku,ijvu->ijkv',queries,keys)
        logits = torch.matmul(queries, keys.permute(0, 1, 3, 2)) # (batch_size, num_heads, queries_seq_len, keys_seq_len)

        weights = F.softmax(logits, dim=-1)

        weights = self.dropout(weights)

        contexts = torch.matmul(weights, values)
        # Merge Heads
        contexts = self._merge_heads(contexts)
        outputs = self.output_linear(contexts)
        return outputs
