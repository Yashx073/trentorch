# Precision, Recall, F1, ROC, AUC

Intermediate | classical-ml | evaluation | classification-metrics

### The problem, from first principles

`01-splitting-and-resampling` gives a clean way to get held-out predictions; it doesn't say how to judge them. Plain accuracy hides a lot: on a dataset that's 95% negative, predicting "negative" for everything scores 95% accuracy while catching zero positives. A metric that only counts "right vs. wrong" can't tell that apart from a genuinely useful classifier.

What's actually needed is a way to look at the _kinds_ of mistakes a classifier makes — how many real positives it misses, how many negatives it wrongly flags — and, since most classifiers output a score rather than a hard `0`/`1`, a way to judge that whole range of possible decision thresholds at once rather than locking in just one.

### From theory to code

Implement `precision_recall_f1(labels, predictions)` for a fixed set of hard predictions, then `roc_curve(labels, scores)` and `auc(fpr, tpr)` for the threshold-swept view built from continuous scores. Theory below derives the confusion-matrix counts and the curve; here the job is just counting matches with boolean masks and applying the resulting formulas.

### Constraints

- `labels`, `predictions` in `precision_recall_f1`: `{0, 1}`-valued, same shape.
- Returns `(precision, recall, f1)`, each `0.0` if its denominator would be `0` (no positive predictions for precision, no actual positives for recall) rather than raising or returning `nan`.
- `scores` in `roc_curve`: continuous, higher means "more likely class 1."
- `roc_curve` returns `(fpr, tpr, thresholds)`, one entry per threshold tried: every distinct score value from highest to lowest, plus `+inf` (the threshold that classifies everything as `0`).
- `tpr`/`fpr` at each threshold are computed against the _total_ actual positive/negative counts, fixed for the whole dataset — not against the current threshold's predicted counts.
- `auc(fpr, tpr)` must not assume `fpr` already arrives sorted ascending.

### Hints

Open one at a time. Each gives away a little more than the last.

<details>
<summary>Hint 1</summary>

`precision_recall_f1` only needs three boolean-mask counts: predicted 1 and actually 1, predicted 1 and actually 0, predicted 0 and actually 1. Everything else is arithmetic on those three numbers, with a zero-denominator guard.

</details>

<details>
<summary>Hint 2</summary>

For `roc_curve`, sweep every distinct score as a threshold, from highest to lowest, plus `+inf` at the start. At each threshold, `predictions = (scores >= threshold)`, then reuse the same TP/FP counting idea as `precision_recall_f1` — but divide by the fixed total positive/negative counts, not the counts among current predictions.

</details>

<details>
<summary>Hint 3</summary>

`roc_curve`'s natural order is descending by threshold, which isn't necessarily ascending `fpr`. `auc` needs to sort both arrays by `fpr` first, then integrate with the trapezoidal rule (`np.trapezoid`). Many points share the same `fpr`, so the sort has to be stable (`kind="stable"`) to keep them in order.

</details>
