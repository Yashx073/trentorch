# Train/test split, k-fold cross-validation, bootstrapping

Beginner | classical-ml | evaluation | resampling

### The problem, from first principles

Every model trained so far in this curriculum has been trained and then, implicitly, trusted. But a model that memorizes its training set can report a perfect training score while being useless on anything new. Before a score means anything, the data used to compute it has to be data the model never touched during training.

The simplest way to guarantee that: cut the dataset into two disjoint pieces before training ever starts, one for fitting the model, one held back purely for judging it. That's a train/test split, and it's the foundation every fancier evaluation scheme in this track builds on.

### From theory to code

Implement `train_test_split(input, labels, test_size, seed=None)` and `k_fold_split(n_samples, k, seed=None)`. Both rely on the same primitive: shuffle sample indices once, then partition the shuffled order into the pieces each function needs.

### Constraints

- `train_test_split` returns `(train_input, test_input, train_labels, test_labels)`.
- `test_size` is a fraction of samples (`0` to `1`) to hold out as the test set; the test set size is `round(n_samples * test_size)`.
- Train and test sets must be disjoint and together cover every original index exactly once.
- `input[i]` and `labels[i]` must stay paired after the split.
- Same `seed` must reproduce the exact same split.
- `k_fold_split` returns a list of `k` `(train_idx, val_idx)` pairs, one per fold acting as validation.
- Every index `0..n_samples-1` appears in exactly one fold's `val_idx` across the `k` pairs.
- Within a pair, `train_idx` and `val_idx` are disjoint and together cover every index.
- Folds must be as close to equal size as `n_samples` and `k` allow, even when `k` doesn't evenly divide `n_samples`.
- No fabricated bootstrapping here — `02-bagging`'s `bootstrap_sample` already implements it; this question doesn't reimplement it.

### Hints

Open one at a time. Each gives away a little more than the last.

<details>
<summary>Hint 1</summary>

Both functions start the same way: shuffle indices once with `rng.permutation(n_samples)`. Everything after that is just slicing or grouping that one shuffled array — never re-shuffling per fold.

</details>

<details>
<summary>Hint 2</summary>

For `k_fold_split`, `np.array_split(shuffled_idx, k)` handles an uneven `n_samples / k` for you. For fold `i`, the validation set is `folds[i]`; the training set is every _other_ fold concatenated together.

</details>
