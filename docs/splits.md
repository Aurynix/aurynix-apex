# Data Splits

> Stage 6 (step 2.2). Code: [`src/apex/data/split.py`](../src/apex/data/split.py). Settings: `config.json → split`.

```bash
make split   # creates data/splits.csv (Prospect ID → train / val / test)
```

---

## The split

| Split | Rows | Share | Conversion rate | Used for |
|---|---|---|---|---|
| **train** | 5,544 | 60% | 38.55% | Fitting the pipeline and the model; cross-validation (steps 2.3–2.5) |
| **val** | 1,848 | 20% | 38.53% | Comparing models, tuning, calibration, choosing thresholds |
| **test** | 1,848 | 20% | 38.53% | **One** final evaluation (step 3.1) |

- **Stratified** on `Converted`, so every split has the same conversion rate (38.5%, within 0.02 points).
- **Seeded** with `random_state = 42`. `test_size` and `val_size` (0.2 each) are shares of all rows.
- **Two steps:** first 20% → test, then 25% of the rest (= 20% of all) → val.

## Saved, not recomputed

The split is saved as `data/splits.csv` (`Prospect ID`, `split`). Every later step reads that file with `load_splits()`, so the split never changes, even if a library update changes how shuffling works. Running `make split` twice gives the same file (MD5 `f22420a3…`). The file is git-ignored like the rest of `data/`; `make split` re-creates it from `Leads.csv` (pinned by its SHA-256 in [data_dictionary.md](data_dictionary.md)).

## The test set rule

The test set is used **once**, in step 3.1. To make that hard to break by accident:

```python
from apex.data.split import load_splits

parts = load_splits()                     # {"train", "val"} only
parts = load_splits(include_test=True)    # step 3.1 only
```

`load_splits()` also raises an error if any lead has no saved split, so a changed dataset cannot silently mix into the splits.

## Why a random split (not by time)

The dataset has no dates. `Lead Number` might reflect arrival order, but nothing documents that ([data_quality.md](data_quality.md) section 6), so a time-based split is not possible. **Risk:** if lead behaviour changes over time, random-split scores may be a little optimistic; drift monitoring (step 4.4) watches for this.

## Look-alike leads across splits

After cleaning, **22% of val and test leads have an identical-looking lead in train** (404 val, 415 test). These are different people with very little information: no web activity and empty form fields (see [data_cleaning.md](data_cleaning.md) section 3).

**Decision: keep the random split.**
- They convert at the normal rate (38.4% val, 39.0% test), so the model gets no shortcut from them.
- New leads in production will look like old ones in the same way, so this reflects real use.
