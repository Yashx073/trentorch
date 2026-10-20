import numpy as np


def train_test_split(
    input: np.ndarray, labels: np.ndarray, test_size: float, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    test_size: fraction of samples (0 to 1) to hold out as the test set.

    Returns:
        (train_input, test_input, train_labels, test_labels)
    """
    # TODO: Shuffle sample indices once (rng.permutation), take the
    # first round(n_samples * test_size) as the test set, the rest as
    # train.
    n_samples = len(input)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)

    n_tests = round(n_samples * test_size)

    train_idx = indices[n_tests:]
    test_idx = indices[:n_tests]

    train_input = input[train_idx]
    test_input = input[test_idx]
    train_labels = labels[train_idx]
    test_labels = labels[test_idx]

    return train_input, test_input, train_labels, test_labels
    
    


def k_fold_split(
    n_samples: int, k: int, seed: int | None = None
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Returns:
        a list of k (train_idx, val_idx) pairs. Each val_idx is one of
        k roughly-equal, non-overlapping folds of shuffled indices
        0..n_samples-1; the matching train_idx is every OTHER fold's
        indices concatenated together. Every sample appears in exactly
        one val_idx across the k pairs.
    """
    # TODO: Shuffle indices once, np.array_split into k folds. For each
    # fold i: val_idx = folds[i], train_idx = every other fold
    # concatenated.
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)

    folds = np.array_split(indices, k)

    splits = []

    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate(folds[:i] + folds[i+1:])

        splits.append((train_idx, val_idx))

    return splits
