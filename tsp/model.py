import numpy as np

import torch
from torch.nn import Linear, LSTM, Parameter
from torch.distributions import Categorical

class Attention(torch.nn.Module):
    def __init__(self, hidden_size, C=10):
        super(Attention, self).__init__()
        self.C = C
        self.W_q = None
        self.W_ref = None
        self.v = None

        self.W_q = Linear(hidden_size, hidden_size)
        self.W_ref = Linear(hidden_size, hidden_size)
        self.v = Linear(hidden_size, 1)
        ########################################################################

    def forward(self, query, target, mask):
        """
        Compute the output of Bahanadu Attention.
        
        INPUT:
            [query] is a tensor of size [batch_size x hidden_size], where
                  query[i, :] is the query vector for the ith batch
            [target] is a tensor of size [batch_size x seq_len x hidden_size],
                  where target[i, j, :] is the jth reference vector of the
                  ith batch
            [mask] is a tensor of size [batch_size x seq_len]. mask[i, j] is
                  True if city j has already been visited in batch i, and is
                  False otherwise
            [M] is a large negative number
        
        OUTPUT:
            [logits] is a tensor of size [batch_size x seq_len]. logits[i, j]
                  is the value at the jth index of the u vector in Bahanadu
                  Attention for batch i. If city j has already been visited
                  in batch i, then logits[i, j] = M.
        """
        _, seq_len, _ = target.shape
        M = -100000.0
        logits = None

        logits = self.v(torch.tanh(self.W_ref(target) + self.W_q(query).unsqueeze(1))).squeeze(2)
        logits[mask] = M

        # Scaling affects exploration
        logits = self.C * torch.tanh(logits)
        return target, logits


class RNNTSP(torch.nn.Module):
    def __init__(
        self,
        embedding_size,
        hidden_size,
        seq_len,
        n_glimpses,
        tanh_exploration,
        force_prob,
        heuristic,
    ):
        super(RNNTSP, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_glimpses = n_glimpses
        self.seq_len = seq_len

        self.force_prob = force_prob
        self.heuristic = heuristic

        self.embedding = Linear(2, embedding_size)
        self.encoder = LSTM(hidden_size, hidden_size, batch_first=True)
        self.decoder = LSTM(hidden_size, hidden_size, batch_first=True)
        self.pointer = Attention(hidden_size, C=tanh_exploration)
        self.ins_ptr = Attention(hidden_size, C=tanh_exploration)

        self.glimpse = Attention(hidden_size)
        self.decoder_start_input = Parameter(torch.FloatTensor(embedding_size))
        self.decoder_start_input.data.uniform_(
            -(1. / np.sqrt(embedding_size)),
            1. / np.sqrt(embedding_size)
        )
        
    def glimpses(self, query, encoder_outputs, mask):
      for _ in range(self.n_glimpses):
          ref, logits = self.glimpse(query, encoder_outputs, mask)
          query = torch.matmul(
              ref.transpose(-1, -2),
              torch.softmax(logits, dim=-1).unsqueeze(-1)
          ).squeeze(-1)

          return query

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]
        prev_chosen_logprobs = []
        prev_chosen_insprobs = []
        prev_chosen_indices = []
        prev_chosen_inserts = []

        def _encode(inputs):
            """
            Compute embeddings for each 2D point using a shared linear layer, then
                apply the encoder LSTM to get a latent memory state for each point
            
            INPUT:
                A tensor [inputs] of size [batch_size x seq_len x 2] of raw
                    Euclidean TSP input.
            
            OUTPUT:
                A tensor [embedded] of size [batch_size x seq_len x
                    self.embedding_size] of embedded TSP input
            
                A tuple [encoder_ouputs, (hidden, context)] of output from the
                    encoder LSTM
            """
            embedded = self.embedding(inputs)
            encoder_outputs, (hidden, context) = self.encoder(embedded)
            ########################################################################

            return embedded, encoder_outputs, hidden, context
        
        self.embedded, self.encoder_outputs, self.hidden, self.context = _encode(inputs)

        # For each batch, all points initially available
        mask = torch.zeros(batch_size, self.seq_len, dtype=torch.bool)
        pos = torch.ones(batch_size, self.seq_len, dtype=torch.bool)

        # Initial decoder input vector
        decoder_input = self.decoder_start_input \
            .unsqueeze(0) \
            .repeat(batch_size, 1)

        # Iterate through decoding decisions
        for i in range(seq_len):

            def _decode(decoder_input, hidden, context):
                """
                Compute a query vector given the current LSTM hidden state,
                    context, and decoder input.
                
                INPUT:
                    [hidden] is the current LSTM hidden state
                    [context] is the current LSTM context
                    [decoder_input] is a tensor of size [batch_size x
                        embedding_size]
                
                OUTPUT:
                    [query] is a tensor of size [batch_size x hidden_size]
                        representing decoder output for the current step
                    [new_hidden] is the updated LSTM hidden state
                    [new_context] is the updated LSTM context
                """
                query, (new_hidden, new_context) = self.decoder(decoder_input.unsqueeze(1), (hidden, context))
                query = query.squeeze(1)
                ####################################################################

                return query, new_hidden, new_context
            
            self.query, self.hidden, self.context = _decode(decoder_input, self.hidden, self.context)
            self.query = self.glimpses(self.query, self.encoder_outputs, mask=mask.clone())
  
            def _attend(query, encoder_outputs, mask, pointer):
                """
                Compute and sample from a distribution over cities that have not
                    yet been visited.
                
                INPUT:
                    [query] is a tensor of size [batch_size x hidden_size]
                        representing decoder output for the current step
                    [encoder_outputs] is a tensor of size [batch_size x seq_len x
                        hidden_size] containing the latent embeddings of each city
                    [mask] is a tensor of size [batch_size x seq_len], where
                        mask[i, j] is True if city j has already been visited in
                        batch i, and is False otherwise
                
                OUTPUT:
                    [cat] is a Categorical distribution over the logits from the
                        pointer network, and is used for training
                    [chosen] is a tensor of size [batch_size], where
                        chosen[i] gives the index of the next city to visit in
                        batch i
                """
                target, logits = pointer(query, encoder_outputs, mask)
                probs = torch.softmax(logits, dim=-1)
                cat = Categorical(probs=probs)
                chosen = cat.sample()
                ####################################################################

                return cat, chosen

            pos[i] = False
            self.cat, self.chosen = _attend(self.query, self.encoder_outputs,
                                            mask=mask.clone(), pointer=self.pointer)
            self.ins, self.insert = _attend(self.query, self.encoder_outputs,
                                            mask=pos.clone(), pointer=self.ins_ptr)

            # Mark visited cities, update decoder input
            log_probs = self.cat.log_prob(self.chosen)
            ins_probs = self.ins.log_prob(self.insert)
            prev_chosen_logprobs.append(log_probs)
            prev_chosen_insprobs.append(ins_probs)

            # Force heuristic
            if torch.rand(1) < self.force_prob and len(prev_chosen_indices) > 0:
                self.chosen, self.insert = self.heuristic(inputs, torch.stack(prev_chosen_indices, 1))

            # Append chosen city
            prev_chosen_indices.append(self.chosen)
            prev_chosen_inserts.append(self.insert)
            mask[[i for i in range(batch_size)], self.chosen] = True
            decoder_input = self.embedded.gather(1, self.chosen[:, None, None].repeat(1, 1, self.hidden_size)).squeeze(1)

        return (torch.stack(prev_chosen_logprobs, 1), torch.stack(prev_chosen_insprobs, 1)), \
               (torch.stack(prev_chosen_indices, 1), torch.stack(prev_chosen_inserts, 1))
