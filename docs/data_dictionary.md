# Data Dictionary & Leakage Audit

> Sources are recorded in stage 2 (step 1.2). The per-column dictionary and leakage audit are completed in stage 4 (step 1.4).

---

## 1. Sources

### 1.1 Primary: Lead Scoring Dataset (X Education)

| Field | Value |
|---|---|
| Publisher | X Education (online education company), published on Kaggle |
| How to find | Kaggle search: *"Lead Scoring X Education"* |
| Kaggle URL | _TBD: fill in the exact dataset page used_ |
| License / terms | _TBD: copy from the Kaggle dataset page_ |
| File | `Leads.csv` → `data/raw/Leads.csv` (git-ignored) |
| Downloaded on | _TBD_ |
| SHA-256 | _TBD: from `make data-info`_ |
| Size | _TBD_ |
| Rows × columns | _TBD_ (expected about 9,000 rows) |
| Target | `Converted` (1 = converted, 0 = not converted) |
| Conversion rate | _TBD_ (expected about 38%) |

The SHA-256 hash pins the exact file. If a re-download gives a different hash, the dataset changed and every downstream result must be re-checked.

To fill in the _TBD_ values, run:

```bash
make data-info
```

### 1.2 Secondary (planned): Bank Marketing (UCI, id 222)

Not yet downloaded. It will be recorded here when the pipeline is applied to it (Phase 5).

---

## 2. Column dictionary & leakage audit

> To be completed in step 1.4.

For every column: meaning, type, missing rate (including `"Select"`), and the **prediction-moment test**: *is this known when the lead enters the SDR queue, before any sales contact?* (see [problem_framing.md](problem_framing.md)).

| Column | Type | Meaning | Missing % | Available at prediction time? | Decision | Reason |
|---|---|---|---|---|---|---|
| _to be filled in step 1.4_ | | | | | | |
