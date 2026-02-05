from itertools import product, islice
import numpy as np
import pandas as pd
import torch
import tqdm.auto as tqdm

from crn_lyapunov.crn import get_drift

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def get_grid_chunks(ranges, chunk_size):
    dim_ranges = [torch.arange(0, r, device=device).float() for r in ranges]

    grid_gen = product(*[r.tolist() for r in dim_ranges])

    while True:
        chunk = list(islice(grid_gen, chunk_size))
        if not chunk:
            break
        yield torch.tensor(chunk, device=device)


def performance_table(
    model,
    net,
    ref_g,
    ranges,
    max_drift_ref=None,
    max_drift_aug=None,
    min_eps=-4,
    chunk_size=1_000_000,
):
    total_states = np.prod(ranges)
    epsilons = torch.logspace(min_eps, 0, 20).to(device)

    max_d_aug = -float("inf")
    max_d_ref = -float("inf")

    if max_drift_ref is None or max_drift_aug is None:
        pbar1 = tqdm.tqdm(total=total_states, desc="Pass 1/2: Finding D_max")
        with torch.no_grad():
            for chunk in get_grid_chunks(ranges, chunk_size):
                if max_drift_aug is None:
                    d_aug = get_drift(model, net, chunk)
                    max_d_aug = max(max_d_aug, d_aug.max().item())
                if max_drift_ref is None:
                    d_ref = get_drift(ref_g, net, chunk)
                    max_d_ref = max(max_d_ref, d_ref.max().item())
                pbar1.update(len(chunk))
        pbar1.close()
    if max_drift_aug is not None:
        max_d_aug = max_drift_aug
    if max_drift_ref is not None:
        max_d_ref = max_drift_ref

    print(f"max D_aug={max_d_aug} max D_ref={max_d_ref}")

    counts_aug = torch.zeros(len(epsilons), device=device)
    counts_ref = torch.zeros(len(epsilons), device=device)

    pbar2 = tqdm.tqdm(total=total_states, desc="Pass 2/2: Counting States")
    with torch.no_grad():
        for chunk in get_grid_chunks(ranges, chunk_size):
            d_aug = get_drift(model, net, chunk).unsqueeze(1)
            d_ref = get_drift(ref_g, net, chunk).unsqueeze(1)
            dmax = d_aug.max()
            if dmax > max_d_aug:
                pbar2.close()
                print(f"Larger max D_aug={dmax} - Restarting...")
                return performance_table(
                    model,
                    net,
                    ref_g,
                    ranges,
                    max_drift_ref=max_drift_ref,
                    max_drift_aug=dmax,
                    min_eps=min_eps,
                    chunk_size=chunk_size,
                )

            counts_aug += (
                (d_aug / max_d_aug * epsilons).squeeze(1) > (epsilons - 1)
            ).sum(0)
            counts_ref += (
                (d_ref / max_d_ref * epsilons).squeeze(1) > (epsilons - 1)
            ).sum(0)

            pbar2.update(len(chunk))
    pbar2.close()

    return pd.DataFrame(
        {
            "epsilon": epsilons.cpu().numpy(),
            "size_aug": counts_aug.cpu().numpy(),
            "size_ref": counts_ref.cpu().numpy(),
        }
    )
