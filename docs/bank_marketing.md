# Bank Marketing: the second dataset

> Phase 6. Goal: run the same Apex pipeline on a very different dataset, to show what is general, and to check that the leakage audit catches `duration` by itself.
> Settings: [`config_bank.json`](../config_bank.json). Every command takes it through `APEX_CONFIG`:

```bash
APEX_CONFIG=config_bank.json make data-download   # UCI zip → data/raw/bank-full.csv
APEX_CONFIG=config_bank.json make data-info       # shape, SHA-256, conversion rate
```

The bank config has its own paths (`models/bank/`, `reports/figures/bank/`, `data/splits_bank.csv`, `data/apex_bank.db`), so it never overwrites the lead model.

| Step | Status |
|---|---|
| B.1 Data collection & quality | ✅ this page, sections 1–2 |
| B.2 Leakage audit (`duration`) | ⏳ |
| B.3 Features | ⏳ |
| B.4 Model, evaluation, segments | ⏳ |
| B.5 Report and comparison with the lead model | ⏳ |

---

## 1. Source

| Field | Value |
|---|---|
| Dataset | *Bank Marketing*, UCI Machine Learning Repository, id 222 (Moro et al., 2011) |
| URL | https://archive.ics.uci.edu/static/public/222/bank+marketing.zip |
| License | CC BY 4.0 (UCI) |
| What it is | Phone marketing campaigns of a Portuguese bank (May 2008 – Nov 2010) to sell a **term deposit** |
| File used | `bank-full.csv` (inside `bank.zip` inside the download) → `data/raw/bank-full.csv`; column descriptions: `bank-names.txt` |
| Not used | `bank-additional-full.csv` (adds 5 economic indicators; more columns, same idea) |
| Downloaded on | 2026-10-06, with `make data-download` |
| SHA-256 (`bank-full.csv`) | `d1513ec63b385506f7cfce9f2c5caa9fe99e7ba4e8c3fa264b3aaf0f849ed32d` |
| Size | 4.61 MB |
| Rows × columns | **45,211 × 17** (+ `row_id` added by the loader) |
| Target | `y`: did the client subscribe? `"yes"` → 1, `"no"` → 0 |
| Conversion rate | **11.7%** (5,289 yes · 39,922 no): much more imbalanced than the leads (38.5%) |

**Unit and prediction moment.** One row = one client contacted in a campaign. The model should score a client **before the call**, so the bank knows whom to call first. (Same idea as for leads: before any sales contact in this campaign.)

### What the loader does differently (all from config)

| Difference | Config | Effect |
|---|---|---|
| Separator `;` | `data.csv_sep` | read correctly |
| Target `"yes"` / `"no"` | `data.target_values` | mapped to 1 / 0; any other value is an error |
| No id column | `data.id_columns = ["row_id"]` | `row_id` = row number 1…45,211, stable as long as the file hash is the same (needed for the saved split) |
| Zip inside a zip | `source.url`, `source.files` | the downloader searches nested zips |

## 2. Columns

From `bank-names.txt`. Conversion rate per value is on all 45,211 rows (11.7% overall).

| Column | Type | Meaning | Notes |
|---|---|---|---|
| `age` | int | Age | ≤ 25: **24%**, 60+: **42%**, 35–60: ~10% |
| `job` | text (12) | Job type | retired 22.8%, management 13.8%, blue-collar 7.3%; `unknown` 0.6% |
| `marital` | text (3) | Marital status | single 14.9%, married 10.1% |
| `education` | text (3) | Education | tertiary 15.0%, primary 8.6%; `unknown` 4.1% |
| `default` | yes/no | Has credit in default | yes 6.4% vs. no 11.8%; only 1.8% "yes" |
| `balance` | int (€) | Average yearly balance | **negative for 8.3%** (overdraft: valid); max 102,127 |
| `housing` | yes/no | Has a housing loan | yes 7.7% vs. no 16.7% |
| `loan` | yes/no | Has a personal loan | yes 6.7% vs. no 12.7% |
| `contact` | text (2) | Contact type of the **last contact** | cellular 14.9%, telephone 13.4%, `unknown` **28.8%** (4.1%) |
| `day`, `month` | int / text | Day and month of the **last contact** | May has 30% of calls but converts at 6.7%; April 19.7% |
| `duration` | int (s) | Length of the **last call** | ⚠️ known only **after** the call; 0 s → always "no" (B.2) |
| `campaign` | int | Contacts in this campaign, **including the last one** | median 2, max 63 |
| `pdays` | int | Days since the last contact in a previous campaign; **−1 = never contacted** | −1 for 81.7% |
| `previous` | int | Contacts before this campaign | median 0, max 275 |
| `poutcome` | text (3) | Result of the previous campaign | success **64.7%**, failure 12.6%; `unknown` 81.7% |
| `y` | target | Subscribed a term deposit | 11.7% |

Several columns describe the **last contact** of this campaign (`contact`, `day`, `month`, `duration`, `campaign`). Which of them are known before the call is the question of step B.2.

## 3. Data quality and cleaning

The **same `clean()` as for the leads** runs on this data; only `config_bank.json` differs. No code was written for this dataset's cleaning.

| # | Issue | Size | Decision | Config |
|---|---|---|---|---|
| 1 | `"unknown"` = hidden missing value | job 0.6%, education 4.1%, contact 28.8%, poutcome 81.7% | → missing, then `"Missing"` category (kept: e.g. contact unknown converts at 4.1%) | `missing_placeholders` |
| 2 | `poutcome = "unknown"` is **not** really missing | 36,954 of its 36,959 rows have `pdays = −1` | It means "no previous campaign"; kept as the `"Missing"` category, which the model learns as that | — |
| 3 | `pdays = −1` as "never" | 81.7% | Valid sentinel; not checked as negative; turned into a clear feature in B.3 | not in `numeric_columns` |
| 4 | Negative `balance` | 3,766 rows (8.3%) | Valid (overdraft); not checked as negative | not in `numeric_columns` |
| 5 | Yes/No columns in lower case | `default`, `housing`, `loan` | → 1 / 0 (encoding now ignores letter case) | `binary_columns` |
| 6 | Outliers | `balance` max 102,127; `campaign` max 63; `previous` max 275 | Cap near the 99.9th percentile: balance 35,000 (41 rows), campaign 30 (59), previous 25 (28). Rows are kept | `caps` |
| 7 | Duplicates | 0 identical rows | Nothing to do | — |
| 8 | Missing values (`NaN`) | none in the file | — | — |
| 9 | Imbalance | 11.7% positive | PR-AUC baseline is 0.117; class weights get tested again in B.4 | — |

**Result:** `clean(load_raw())` → 45,211 rows × 17 columns, 0 missing values, no rows removed.

## 4. Leakage audit (B.2)

⏳

## 5. Features (B.3)

⏳

## 6. Model and results (B.4)

⏳
