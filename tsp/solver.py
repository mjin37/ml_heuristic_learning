import torch
import torch.nn as nn
from torch.autograd import Variable


class Solver(nn.Module):
    def __init__(self, model, *args):
        super(Solver, self).__init__()

        self.actor = model(*args)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def reward(self, sample_solution):
        """
        Args:
            sample_solution seq_len of [batch_size]
            torch.LongTensor [batch_size x seq_len x 2]
        """
        batch_size, seq_len, _ = sample_solution.size()

        tour_len = Variable(torch.zeros([batch_size]))
        if isinstance(sample_solution, torch.cuda.FloatTensor):
            tour_len = tour_len.cuda()
        for i in range(seq_len - 1):
            tour_len += torch.norm(sample_solution[:, i, :] - sample_solution[:, i + 1, :], dim=-1)

        tour_len += torch.norm(sample_solution[:, seq_len - 1, :] - sample_solution[:, 0, :], dim=-1)

        return tour_len

    def forward(self, inputs):
        """
        Args:
            inputs: [batch_size, input_size, seq_len]
        """
        probs, (actions, inserts) = self.actor(inputs)
        actions = self._reorder(actions, inserts)
        R = self.reward(inputs.gather(1, actions.unsqueeze(2).repeat(1, 1, 2)))

        return R, probs, actions

    def _reorder(self, actions, inserts):
        B, S = actions.shape
        output = torch.zeros((B, S), dtype=torch.int64)
        i = torch.arange(B).long().to(self.device)

        def kthvalue(input, k, dim=None):
            sorted, indices = torch.sort(input)
            return sorted[torch.arange(len(k)), k], indices[torch.arange(len(k)), k]

        for i in range(S):
            idx = inserts[:, -i]
            _, kth = kthvalue(output, idx, dim=-1)
            output[i, kth] = actions[:, -1]

        return output
