import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import math
from typing import NamedTuple
from nets.graph_encoder import GraphAttentionEncoder
from nets.ha_graph_encoder import GraphHAttentionEncoder
from torch.nn import DataParallel
from utils.beam_search import CachedLookup
from utils.functions import sample_many
from nets.positional_encoding import PostionalEncoding

def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)
        return AttentionModelFixed(
            node_embeddings=self.node_embeddings[key],
            context_node_projected=self.context_node_projected[key],
            glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
            glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
            logit_key=self.logit_key[key]
        )


class AttentionModel(nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 problem,
                 agent_num = 3,
                 n_encode_layers=2,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization='batch',
                 n_heads=8,
                 checkpoint_encoder=False,
                 shrink_size=None,
                 ft="N"):
        super(AttentionModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0

        self.agent_num = agent_num
        self.ft = ft
        self.tanh_clipping = tanh_clipping

        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        self.problem = problem
        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size
        self.positional_encoding = PostionalEncoding(d_model=embedding_dim, max_len=10000)
        
        # Problem specific context parameters (placeholder and step context dimension)
        step_context_dim = 2*embedding_dim + 2 # Embedding of current_agent, current node, # of left cities and # of left agents
        node_dim = 2  # x, y
        self.init_embed_depot = nn.Linear(2, embedding_dim)
        self.pos_emb_proj = nn.Sequential(nn.Linear(embedding_dim, embedding_dim, bias=False))
        self.alpha = nn.Parameter(torch.Tensor([1]))

        if problem.NAME == "mtsp":
            self.dis_emb =  nn.Sequential(nn.Linear(3, embedding_dim, bias=False))
            
            self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization
        )
        elif problem.NAME == "mpdp":
            self.init_embed_pick = nn.Linear(node_dim * 2, embedding_dim)
            self.init_embed_delivery = nn.Linear(node_dim, embedding_dim)
            self.dis_emb =  nn.Sequential(nn.Linear(5, embedding_dim, bias=False))
            self.embedder = GraphHAttentionEncoder(n_heads=n_heads,
                                                   embed_dim=embedding_dim,
                                                   n_layers=self.n_encode_layers,
                                                   normalization=normalization
                                                   )
            self.embedder.agent_num = agent_num
        # Using the finetuned context Encoder 
        if self.ft == "Y":
            self.contextual_emb = nn.Sequential(nn.Linear(embedding_dim, 8 * embedding_dim, bias=False),
            nn.ReLU(),
            nn.Linear(8 * embedding_dim, embedding_dim, bias=False)
            )


        self.init_embed = nn.Linear(node_dim, embedding_dim)
        
        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_step_context = nn.Linear(step_context_dim, embedding_dim, bias=False)
        assert embedding_dim % n_heads == 0
        # Note n_heads * val_dim == embedding_dim so input to project_out is embedding_dim
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)


    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp


    def forward(self, input, return_pi=False):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :param return_pi: whether to return the output sequences, this is optional as it is not compatible with
        using DataParallel as the results may be of different lengths on different GPUs
        :return:
        """

        if self.checkpoint_encoder and self.training:  # Only checkpoint if we need gradients
            embeddings, _ = checkpoint(self.embedder, self._init_embed(input))
        else:
            embeddings, _ = self.embedder(self._init_embed(input), agent_num=self.agent_num)

        _log_p, pi, cost = self._inner(input, embeddings)
    
        # Log likelyhood is calculated within the model since returning it per action does not work well with
        # DataParallel since sequences can be of different lengths
        ll = self._calc_log_likelihood(_log_p, pi, None)
        
        if return_pi:
            return cost, ll, pi
        
        return cost, ll


    def precompute_fixed(self, input):
        embeddings, _ = self.embedder(self._init_embed(input))
        # Use a CachedLookup such that if we repeatedly index this object with the same index we only need to do
        # the lookup once... this is the case if all elements in the batch have maximum batch size
        return CachedLookup(self._precompute(embeddings))


    def _calc_log_likelihood(self, _log_p, a, mask):

        # Get log_p corresponding to selected actions
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        if mask is not None:
            log_p[mask] = 0

        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        return log_p.sum(1)

    def _init_embed(self, input):

        # mTSP
        
        if self.problem.NAME == "mtsp":
            if len(input.size()) == 2:
                input = input.unsqueeze(0)
            num_cities = input.size(1) - 1
            self.num_cities = num_cities
            
            # Embedding of depot
            depot_embedding = self.init_embed_depot(input[:, 0:1, :])
            
            # Make the depot embedding the same for all agents
            depot_embedding = depot_embedding.repeat(1, self.agent_num + 1, 1)
            
            positional_embedding = self.positional_encoding(depot_embedding.size(0), depot_embedding.size(1))
            positional_embedding = positional_embedding.to(depot_embedding.device)
            positional_embedding = self.alpha * self.pos_emb_proj(positional_embedding) / self.agent_num
            
            # Add the positional embedding to the depot embedding to give order bias to the agents
            depot_embedding = depot_embedding + positional_embedding[None,:,:]

            return torch.cat(
                    (depot_embedding,
                        self.init_embed(input[:, 1:, :])),
                    1
                )
        
        elif self.problem.NAME == "mpdp":
            n_loc = input['loc'].size(1)
            if len(input['depot'].size()) == 2:
                input['depot'] = input['depot'][:,None,:]
            original_depot = input['depot']
            new_input = input['loc']
            new_depot = original_depot.repeat(1, self.agent_num + 1, 1)
            embed_depot = self.init_embed_depot(new_depot)
            self.num_request = n_loc // 2

            positional_embedding = self.positional_encoding(new_depot.size(0), new_depot.size(1))
            positional_embedding = positional_embedding.to(embed_depot.device)
            positional_embedding = self.alpha * self.pos_emb_proj(positional_embedding) / self.agent_num
            feature_pick = torch.cat([new_input[:, :n_loc // 2, :], new_input[:, n_loc // 2:, :]], -1)
            feature_delivery = new_input[:, n_loc // 2:, :]  # [batch_size, graph_size//2, 2]
            embed_pick = self.init_embed_pick(feature_pick)
            embed_delivery = self.init_embed_delivery(feature_delivery)
            embed_depot = embed_depot + positional_embedding[None,:,:] 
            
            return torch.cat([embed_depot, embed_pick, embed_delivery], 1)
        

    def _inner(self, input, embeddings):

        outputs = []
        sequences = []
        
        state = self.problem.make_state(input, self.agent_num)

        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        fixed = self._precompute(embeddings)

        batch_size = state.ids.size(0)
        # Perform decoding steps
        i = 0
        while not (self.shrink_size is None and state.all_finished()):
            if self.shrink_size is not None:
                unfinished = torch.nonzero(state.get_finished() == 0)
                if len(unfinished) == 0:
                    break
                unfinished = unfinished[:, 0]
                # Check if we can shrink by at least shrink_size and if this leaves at least 16
                # (otherwise batch norm will not work well and it is inefficient anyway)
                if 16 <= len(unfinished) <= state.ids.size(0) - self.shrink_size:
                    # Filter states
                    state = state[unfinished]
                    fixed = fixed[unfinished]

            log_p, mask = self._get_log_p(fixed, state)

            # Select the indices of the next nodes in the sequences, result (batch_size) long
            selected = self._select_node(log_p.exp()[:, 0, :], mask[:, 0, :])  # Squeeze out steps dimension
            state = state.update(selected)
            
            # Now make log_p, selected desired output size by 'unshrinking'
            if self.shrink_size is not None and state.ids.size(0) < batch_size:
                log_p_, selected_ = log_p, selected
                log_p = log_p_.new_zeros(batch_size, *log_p_.size()[1:])
                selected = selected_.new_zeros(batch_size)

                log_p[state.ids[:, 0]] = log_p_
                selected[state.ids[:, 0]] = selected_

            # Collect output of step
            outputs.append(log_p[:, 0, :])
            sequences.append(selected)
            i += 1
        cost = torch.max(state.lengths, dim=-1)[0]
        return torch.stack(outputs, 1), torch.stack(sequences, 1), cost


    def sample_many(self, input, batch_rep=1, iter_rep=1, agent_num=3, aug=False):
        """
        :param input: (batch_size, graph_size, node_dim) input node features
        :return:
        """
        # Bit ugly but we need to pass the embeddings as well.
        # Making a tuple will not work with the problem.get_cost function
        return sample_many(
            lambda input: self._inner(*input),  # Need to unpack tuple into arguments
            lambda input, pi: self.problem.get_costs(input[0], pi),  # Don't need embeddings as input to get_costs
            (input, self.embedder(self._init_embed(input),agent_num)[0]),  # Pack input with embeddings (additional input)
            batch_rep, iter_rep, aug
        )

    def _select_node(self, probs, mask):

        assert (probs == probs).all(), "Probs should not contain any nans"

        if self.decode_type == "greedy":
            _, selected = probs.max(1)
            assert not mask.gather(1, selected.unsqueeze(
                -1)).data.any(), "Decode greedy: infeasible action has maximum probability"

        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)

            # Check if sampling went OK, can go wrong due to bug on GPU
            # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print('Sampled bad values, resampling!')
                selected = probs.multinomial(1).squeeze(1)

        else:
            assert False, "Unknown decode type"
        return selected

    def _precompute(self, embeddings, num_steps=1):

        # The fixed context projection of the graph embedding is calculated only once for efficiency
        graph_embed = embeddings.mean(1)
        # fixed context = (batch_size, 1, embed_dim) to make broadcastable with parallel timesteps
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]

        # The projection of the node embeddings for the attention is calculated once up front
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)

        # No need to rearrange key for logit as there is a single head
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous()
        )
        return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)

    def _get_log_p_topk(self, fixed, state, k=None, normalize=True):
        log_p, _ = self._get_log_p(fixed, state, normalize=normalize)

        # Return topk
        if k is not None and k < log_p.size(-1):
            return log_p.topk(k, -1)

        # Return all, note different from torch.topk this does not give error if less than k elements along dim
        return (
            log_p,
            torch.arange(log_p.size(-1), device=log_p.device, dtype=torch.int64).repeat(log_p.size(0), 1)[:, None, :]
        )

    def _get_log_p(self, fixed, state, normalize=True):
        
        if self.problem.NAME == "mtsp":
            query = fixed.context_node_projected + \
                    self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings, state)) + self.dis_emb(torch.cat((state.lengths.gather(-1, state.count_depot), state.max_distance, state.remain_max_distance),-1))[:,None,:]
        elif self.problem.NAME == "mpdp":
            query = fixed.context_node_projected + \
                self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings, state)) \
                +self.dis_emb(torch.cat((state.lengths.gather(-1, state.count_depot), state.remain_pickup_max_distance, state.remain_delivery_max_distance, state.longest_lengths.gather(-1, state.count_depot), state.remain_sum_paired_distance / (self.agent_num - (state.count_depot))),-1))[:,None,:] 
        
        # Add Finetuned the context node to the query
        if self.ft == "Y":
            context = query.detach()
            query = self.contextual_emb(query)
            query += context
        
        # Compute keys and values for the nodes
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed, state)

        # Compute the mask
        mask = state.get_mask()

        # Compute logits (unnormalized log_p)
        log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask)

        if normalize:
            log_p = torch.log_softmax(log_p / self.temp, dim=-1)

        assert not torch.isnan(log_p).any()

        return log_p, mask

    def _get_parallel_step_context(self, embeddings, state, from_depot=False):
        """
        Returns the context per step, optionally for multiple steps at once (for efficient evaluation of the model)
        
        :param embeddings: (batch_size, graph_size, embed_dim)
        :param prev_a: (batch_size, num_steps)
        :param first_a: Only used when num_steps = 1, action of first step or None if first step
        :return: (batch_size, num_steps, context_dim)
        """

        current_node = state.get_current_node()
        batch_size, num_steps = current_node.size()
        
        if self.problem.NAME == "mtsp":
            return torch.cat((embeddings.gather(
                        1,
                        torch.cat((current_node,  state.agent_idx), 1)[:, :, None].expand(batch_size, 2, embeddings.size(-1))
                    ).view(batch_size, 1, -1),  1.0 - torch.ones(size = state.count_depot[:,:,None].shape, device=embeddings.device) * (state.count_depot[:,:,None]+1) / self.agent_num, 
                    state.left_city[:,:,None]/self.num_cities),2)
        elif self.problem.NAME == "mpdp":
            return torch.cat((embeddings.gather(
                        1,
                        torch.cat((current_node, state.agent_idx), 1)[:, :, None].expand(batch_size, 2, embeddings.size(-1))
                    ).view(batch_size, 1, -1), 1.0 - torch.ones(size = state.count_depot[:,:,None].shape, device=embeddings.device) * (state.count_depot[:,:,None]+1) / self.agent_num, 
                    state.left_request[:,:,None]/self.num_request),2)
        

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):

        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        # Compute the glimpse, rearrange dimensions so the dimensions are (n_heads, batch_size, num_steps, 1, key_size)
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_steps, graph_size)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        if self.mask_inner:
            assert self.mask_logits, "Cannot mask inner without masking logits"
            compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf

        # Batch matrix multiplication to compute heads (n_heads, batch_size, num_steps, val_size)
        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)

        
        # Project to get glimpse/updated context node embedding (batch_size, num_steps, embedding_dim)
        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size))

        final_Q = glimpse
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))
        
        # From the logits compute the probabilities by clipping, masking and softmax
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[mask] = -math.inf

        return logits, glimpse.squeeze(-2)


    def _get_attention_node_data(self, fixed, state):

        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key
    

    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )
