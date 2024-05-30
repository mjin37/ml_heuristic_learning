import torch
import torch.nn as nn
from torch.autograd import Variable


class Solver(nn.Module):
    def __init__(self, model, *args):
        super(Solver, self).__init__()

        self.actor = model(*args)

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
        probs, actions = self.actor(inputs)
        R = self.reward(inputs.gather(1, actions.unsqueeze(2).repeat(1, 1, 2)))

        return R, probs, actions
