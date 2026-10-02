# Data Dictionary & Leakage Audit

> Sources are recorded in stage 2 (step 1.2). The per-column dictionary and leakage audit are completed in stage 4 (step 1.4).

---

## 1. Sources

### 1.1 Primary: Lead Scoring Dataset (X Education)

| Field | Value |
|---|---|
| Dataset | *Lead Scoring Dataset* (X Education, an online education company) |
| Kaggle | [`amritachatterjee09/lead-scoring-dataset`](https://www.kaggle.com/datasets/amritachatterjee09/lead-scoring-dataset), version 2 (last updated 2020-08-17) |
| License | Listed as **Unknown** on Kaggle. Used for research and portfolio purposes only; the data is never redistributed or committed to this repo. |
| Files | `Lead Scoring.csv` → `data/raw/Leads.csv`<br>`Leads Data Dictionary.xlsx` → `data/raw/Leads Data Dictionary.xlsx` (publisher's column descriptions)<br>Both git-ignored |
| Downloaded on | 2026-10-02, with `make data-download` (`kagglehub`) |
| SHA-256 (`Leads.csv`) | `1426dffd94246b8c7f08d1080d9c04115a0c4b67eadcba1ed9d2fa10d3762802` |
| Size | 2.37 MB |
| Rows × columns | **9,240 × 37** |
| Target | `Converted` (1 = converted, 0 = not converted) |
| Conversion rate | **38.54%** (3,561 converted · 5,679 not converted); moderately imbalanced |

The SHA-256 hash pins the exact file. If a re-download gives a different hash, the dataset changed and every downstream result must be re-checked.

To reproduce:

```bash
make data-download   # fetch from Kaggle into data/raw/
make data-info       # print shape, SHA-256, conversion rate, and column list
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
