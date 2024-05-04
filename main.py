#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import math
import torch

from torch.nn import Linear, LSTM, Parameter
from torch.distributions import Categorical

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("PyTorch has version {}".format(torch.__version__))
print('Using device:', device)

from tsp.util import save, load, compute_solution, plot_tsp
from tsp.train import train
from tsp.data import TSPDataset

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')


class Attention(torch.nn.Module):
    def __init__(self, hidden_size, C=10):
        super(Attention, self).__init__()
        self.C = C
        self.W_q = None
        self.W_ref = None
        self.v = None

        ########################################################################
        # TODO: Your code here!
        # Initialize the linear layers [W_q], [W_ref], and [v].
        #
        # INPUT:
        #   [hidden_size] is the dimension of the query and reference vectors.
        #
        # Our implementation is ~3 lines, but feel free to deviate from this.
        #
        self.W_q = Linear(hidden_size, hidden_size)
        self.W_ref = Linear(hidden_size, hidden_size)
        self.v = Linear(hidden_size, 1)
        ########################################################################

    def forward(self, query, target, mask):

        _, seq_len, _ = target.shape
        M = -100000.0
        logits = None

        ########################################################################
        # TODO: Your code here!
        # Compute the output of Bahanadu Attention.
        #
        # INPUT:
        #     [query] is a tensor of size [batch_size x hidden_size], where
        #           query[i, :] is the query vector for the ith batch
        #     [target] is a tensor of size [batch_size x seq_len x hidden_size],
        #           where target[i, j, :] is the jth reference vector of the
        #           ith batch
        #     [mask] is a tensor of size [batch_size x seq_len]. mask[i, j] is
        #           True if city j has already been visited in batch i, and is
        #           False otherwise
        #     [M] is a large negative number
        #
        # OUTPUT:
        #     [logits] is a tensor of size [batch_size x seq_len]. logits[i, j]
        #           is the value at the jth index of the u vector in Bahanadu
        #           Attention for batch i. If city j has already been visited
        #           in batch i, then logits[i, j] = M.
        #
        # Our implementation is ~7 lines, but feel free to deviate from this.
        #
        logits = self.v(torch.tanh(self.W_ref(target) + self.W_q(query).unsqueeze(1))).squeeze(2)
        logits[mask] = M
        ########################################################################

        # Scaling affects exploration, feel free to ignore this.
        logits = self.C * torch.tanh(logits)
        return target, logits


class RNNTSP(torch.nn.Module):
    def __init__(
        self,
        embedding_size,
        hidden_size,
        seq_len,
        n_glimpses,
        tanh_exploration
    ):
        super(RNNTSP, self).__init__()


        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_glimpses = n_glimpses
        self.seq_len = seq_len

        self.embedding = None
        self.encoder = None
        self.decoder = None
        self.pointer = None

        ########################################################################
        # TODO: Your code here!
        # Initialize the linear embedding layer, encoder and decoder LSTMs, and
        #     pointer network.
        #
        # INPUT:
        #     [embedding_size] is the size of the city embeddings X_i
        #     [hidden_size] is the dimension of the LSTM hidden states
        #
        # Our solution is ~4 lines, but feel free to deviate from this.
        #
        # NOTE: For any LSTMs, you must specify the argument "batch_first=True"!
        #
        self.embedding = Linear(2, embedding_size)
        self.encoder = LSTM(hidden_size, hidden_size, batch_first=True)
        self.decoder = LSTM(hidden_size, hidden_size, batch_first=True)
        self.pointer = Attention(hidden_size)
        ########################################################################

        self.glimpse = Attention(hidden_size)
        self.decoder_start_input = Parameter(torch.FloatTensor(embedding_size))
        self.decoder_start_input.data.uniform_(
            -(1. / math.sqrt(embedding_size)),
            1. / math.sqrt(embedding_size)
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
        prev_chosen_indices = []

        def _encode(inputs):
            embedded = None
            encoder_outputs, (hidden, context) = None, (None, None)

            ########################################################################
            # TODO: Your code here!
            # Compute embeddings for each 2D point using a shared linear layer, then
            #     apply the encoder LSTM to get a latent memory state for each point
            #
            # INPUT:
            #     A tensor [inputs] of size [batch_size x seq_len x 2] of raw
            #         Euclidean TSP input.
            #
            # OUTPUT:
            #     A tensor [embedded] of size [batch_size x seq_len x
            #         self.embedding_size] of embedded TSP input
            #
            #     A tuple [encoder_ouputs, (hidden, context)] of output from the
            #         encoder LSTM
            #
            # Our solution is 2 lines.
            #
            embedded = self.embedding(inputs)
            encoder_outputs, (hidden, context) = self.encoder(embedded)
            ########################################################################

            return embedded, encoder_outputs, hidden, context
        
        self.embedded, self.encoder_outputs, self.hidden, self.context = _encode(inputs)

        # For each batch, all points initially available
        mask = torch.zeros(batch_size, self.seq_len, dtype=torch.bool)

        # Initial decoder input vector
        decoder_input = self.decoder_start_input \
            .unsqueeze(0) \
            .repeat(batch_size, 1)

        # Iterate through decoding decisions
        for _ in range(seq_len):

            def _decode(decoder_input, hidden, context):

                query = None
                new_hidden = None
                new_context = None

                ####################################################################
                # TODO: Your code here!
                # Compute a query vector given the current LSTM hidden state,
                #     context, and decoder input.
                #
                # INPUT:
                #     [hidden] is the current LSTM hidden state
                #     [context] is the current LSTM context
                #     [decoder_input] is a tensor of size [batch_size x
                #         embedding_size]
                #
                # OUTPUT:
                #     [query] is a tensor of size [batch_size x hidden_size]
                #         representing decoder output for the current step
                #     [new_hidden] is the updated LSTM hidden state
                #     [new_context] is the updated LSTM context
                #
                # Our solution is ~2 lines.
                #
                # HINT: any input to an LSTM must be of the shape
                #     (batch_size, seq_length, embedding_dimension)
                #
                # What is the sequence length here?
                #
                query, (new_hidden, new_context) = self.decoder(decoder_input.unsqueeze(1), (hidden, context))
                query = query.squeeze(1)
                ####################################################################

                return query, new_hidden, new_context
            
            self.query, self.hidden, self.context = _decode(decoder_input, self.hidden, self.context)
            self.query = self.glimpses(self.query, self.encoder_outputs, mask=mask.clone())
  
            def _attend(query, encoder_outputs, mask):
                chosen = None
                cat = None

                ####################################################################
                # TODO: Your code here!
                # Compute and sample from a distribution over cities that have not
                #     yet been visited.
                #
                # INPUT:
                #     [query] is a tensor of size [batch_size x hidden_size]
                #         representing decoder output for the current step
                #     [encoder_outputs] is a tensor of size [batch_size x seq_len x
                #         hidden_size] containing the latent embeddings of each city
                #     [mask] is a tensor of size [batch_size x seq_len], where
                #         mask[i, j] is True if city j has already been visited in
                #         batch i, and is False otherwise
                #
                # OUTPUT:
                #     [cat] is a Categorical distribution over the logits from the
                #         pointer network, and is used for training
                #     [chosen] is a tensor of size [batch_size], where
                #         chosen[i] gives the index of the next city to visit in
                #         batch i
                #
                # Our solution is ~4 lines.
                # Steps:
                #     1. compute logits from the pointer network
                #     2. apply a softmax to the logits to get probabilites
                #     3. initialize and sample from a Categorical distribution
                #         (https://pytorch.org/docs/stable/distributions.html)
                #
                target, logits = self.pointer(query, encoder_outputs, mask)
                probs = torch.softmax(logits, dim=-1)
                cat = Categorical(probs=probs)
                chosen = cat.sample()
                ####################################################################

                return cat, chosen

            self.cat, self.chosen = _attend(self.query, self.encoder_outputs, mask=mask.clone())

            # Mark visited cities, update decoder input
            mask[[i for i in range(batch_size)], self.chosen] = True
            log_probs = self.cat.log_prob(self.chosen)
            decoder_input = self.embedded.gather(1, self.chosen[:, None, None].repeat(1, 1, self.hidden_size)).squeeze(1)
            prev_chosen_logprobs.append(log_probs)
            prev_chosen_indices.append(self.chosen)

        return torch.stack(prev_chosen_logprobs, 1), torch.stack(prev_chosen_indices, 1)


class TSPArgs(argparse.Namespace):
    def __init__(self):
        self.model = RNNTSP
        self.seq_len = 30
        self.num_epochs = 100
        self.num_tr_dataset = 10 # 10000
        self.num_te_dataset = 20 # 2000
        self.embedding_size = 128
        self.hidden_size = 128
        self.batch_size = 64
        self.grad_clip = 1.5
        self.use_cuda = False # True
        self.beta = 0.9

args = TSPArgs()
train_dataset = TSPDataset()
test_dataset = TSPDataset()
train_dataset.random_fill(args.seq_len, args.num_tr_dataset)
test_dataset.random_fill(args.seq_len, args.num_te_dataset)

model = train(train_dataset, test_dataset, args)
save(model, args, "pre_trained_rnn")


def normalize(embeddings):
    shifted = embeddings - torch.min(embeddings, dim=0).values
    return shifted / torch.max(shifted, dim=0).values

# # Generate input
# usa_tour = TSPDataset()
# 
# usa_tour.tensor_fill(normalize(capitals))
# 
# # Apply the trained model
# model, args = load(RNNTSP, "pre_trained_rnn", device='cpu')
# path = compute_solution(model, usa_tour)
# 
# # Plot the resulting tour
# plot_tsp(capitals, path)
