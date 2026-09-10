import numpy as np



def has_mask_run(arr, mask_value=16, min_run_length=20):
    """Return True if arr contains a run of mask_value 
    with length >= min_run_length.
    """
    arr = np.asarray(arr)
    is_mask = arr == mask_value

    if not np.any(is_mask):
        return False

    diff = np.diff(is_mask.astype(int))
    run_starts = np.where(diff == 1)[0] + 1
    run_ends   = np.where(diff == -1)[0] + 1

    if is_mask[0]:
        run_starts = np.r_[0, run_starts]
    if is_mask[-1]:
        run_ends = np.r_[run_ends, len(arr)]

    run_lengths = run_ends - run_starts
    return np.any(run_lengths >= min_run_length)




def collapse_mask_runs(masked_gene, sequences, mask_value=16, min_run_length=1):
    """Collapse runs of mask_value in masked_gene with 
    length >= min_run_length.

    sequences: dict of {name: array}
    masked_gene must be one of the sequences.
    """
    masked_gene = np.asarray(masked_gene)
    seqs = {k: np.asarray(v).copy() for k, v in sequences.items()}

    is_mask = masked_gene == mask_value

    diff = np.diff(is_mask.astype(int))
    run_starts = np.where(diff == 1)[0] + 1
    run_ends   = np.where(diff == -1)[0] + 1

    if is_mask[0]:
        run_starts = np.r_[0, run_starts]
    if is_mask[-1]:
        run_ends = np.r_[run_ends, len(masked_gene)]

    keep = np.ones(len(masked_gene), dtype=bool)

    for start, end in zip(run_starts, run_ends):
        length = end - start

        if length >= min_run_length:
            keep[start + 1:end] = False

            for name in seqs:
                seqs[name][start] = mask_value

    collapsed = {k: v[keep] for k, v in seqs.items()}
    return collapsed