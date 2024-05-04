import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .solver import solver_RNN
from .tsp_heuristic import get_ref_reward


def train(train_dataset, test_dataset, args):
    if args.use_cuda:
        use_pin_memory = True
    else:
        use_pin_memory = False


    train_data_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=use_pin_memory)

    eval_loader = DataLoader(test_dataset, batch_size=args.num_te_dataset, shuffle=False)

    # Calculating heuristics
    heuristic_distance = torch.zeros(args.num_te_dataset)
    for i, pointset in tqdm(test_dataset):
        heuristic_distance[i] = get_ref_reward(pointset)

    model = solver_RNN(
        args.model,
        args.embedding_size,
        args.hidden_size,
        args.seq_len,
        2,
        10
    )

    if args.use_cuda:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=3.0 * 1e-4)

    # Train loop
    best_model = None
    best_approx_ratio = 10000
    moving_avg = torch.zeros(args.num_tr_dataset)
    if args.use_cuda:
        moving_avg = moving_avg.cuda()

    # Generating first baseline
    for (indices, sample_batch) in tqdm(train_data_loader):
        if args.use_cuda:
            sample_batch = sample_batch.cuda()
        rewards, _, _ = model(sample_batch)
        moving_avg[indices] = rewards

    # Training
    model.train()
    for epoch in range(args.num_epochs):
        for batch_idx, (indices, sample_batch) in enumerate(train_data_loader):
            if args.use_cuda:
                sample_batch = sample_batch.cuda()
            rewards, log_probs, action = model(sample_batch)
            moving_avg[indices] = moving_avg[indices] * args.beta + rewards * (1.0 - args.beta)
            advantage = rewards - moving_avg[indices]
            log_probs = torch.sum(log_probs, dim=-1)
            log_probs[log_probs < -100] = -100
            loss = (advantage * log_probs).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        model.eval()
        ret = []
        for i, batch in eval_loader:
            if args.use_cuda:
                batch = batch.cuda()
            R, _, _ = model(batch)

        if args.use_cuda:
            R = R.cpu()

        approximation_ratio = (R / heuristic_distance).mean().detach().numpy()
        print(f"[At epoch {epoch}] RL model approximation: {approximation_ratio}")
        print("AVG R", R.mean().detach().numpy())

        if approximation_ratio < best_approx_ratio:
            best_model = model
        model.train()
    
    return best_model
