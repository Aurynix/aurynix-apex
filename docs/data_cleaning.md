# Data Cleaning

> What `clean()` does to the raw leads, and why. Code: [`src/apex/data/clean.py`](../src/apex/data/clean.py). Rules and values: `config.json → data`.
> Findings behind these rules: [data_quality.md](data_quality.md) (step 1.3) and [data_dictionary.md](data_dictionary.md) (leakage audit, step 1.4).

```bash
make preprocess   # data/raw/Leads.csv → data/interim/leads_clean.parquet
```

**Before:** 9,240 rows × 37 columns, hidden nulls, text Yes/No, outliers, post-contact columns.
**After:** 9,240 rows × 13 columns (12 features + `Converted`), **0 missing values**, correct types, no leakage. **No rows are removed.**

---

## How it works

One function, `clean(df)`, runs 10 small steps in a fixed order. Every rule and every value (fill values, caps, kept categories) is **fixed in `config.json`**, so cleaning learns nothing at run time. That is why the same function can clean the training file **and** one new lead in the API, with the same result.

| # | Step | Function | What it does |
|---|---|---|---|
| 1 | Normalize text | `normalize_text` | Strip spaces, collapse double spaces |
| 2 | Hidden nulls | `replace_placeholders` | `"Select"`, `"unknown"` → missing |
| 3 | Spelling | `apply_aliases` | `google` → `Google` |
| 4 | Yes/No | `encode_binary` | `"Yes"`/`"No"` → `1`/`0`; any other value is an error |
| 5 | Drop columns | `clean` | IDs, constant columns, empty column, leakage columns |
| 6 | Group rare values | `group_other` | `Country`: India / Other |
| 7 | Validate | `validate` | Target is 0/1, no negative numbers, no logic errors |
| 8 | Fill missing | `fill_missing` | Text → `"Missing"`, numbers → fixed values |
| 9 | Fix types | `clean` | Visit counts and time → integers |
| 10 | Cap outliers | `cap_outliers` | Clip extreme values to a fixed limit |

---

## 1. Missing values

| Column | Missing | Decision | Why |
|---|---|---|---|
| `Specialization`, `City`, `Country`, occupation, "what matters most", `Lead Source` | 0.4% – 39.7% | **Fill with `"Missing"`** | Missingness is a signal: leads with no occupation convert at 13.8% vs. 48.7%. Dropping rows would lose 40%+ of the data; guessing a value would hide the signal |
| `How did you hear about X Education` | **78.5%** | **Drop column** | Almost empty, and weak: conversion is 37% when missing and 22%–50% across tiny groups when present |
| `TotalVisits`, `Page Views Per Visit` | 1.5% (same 137 rows) | **Fill with 3 and 2** (dataset medians) | These leads *did* spend time on the site (11 to 2,217 seconds), so `0` would be wrong. Visits simply were not tracked (110 of 137 are `Lead Add Form`) |

`"Select"` is the default option of the form drop-downs, so it is a missing value too (step 2).

## 2. Outliers

| Column | Median | 99th pct | Max | Cap | Rows capped |
|---|---|---|---|---|---|
| `TotalVisits` | 3 | 17 | **251** | **30** | 10 |
| `Page Views Per Visit` | 2 | 9 | **55** | **15** | 5 |
| `Total Time Spent on Website` | 248 | 1,841 | 2,272 | none | 0 |

**Decision:** clip, do not delete. Values like 251 visits are probably bots or very heavy users. They are real rows with a real outcome, but one extreme value should not pull a linear model. The caps are round numbers near the 99.9th percentile, so only 0.1% of rows are touched. Time on site has no long tail, so it is not capped.

## 3. Duplicates

- **Duplicate IDs:** 0. **Fully duplicated rows (with IDs):** 0.
- **Rows identical once IDs are removed:** 1,953 of 9,240 after cleaning. These are mostly zero-activity API leads with empty optional fields: different people who look the same in this data. 35 rows sit in groups where identical leads have **different outcomes**.
- **Decision: keep them.** Removing them would change the real mix of incoming leads that the model sees in production.

## 4. Leakage

9 columns are dropped because they are filled in **after** a sales rep contacts the lead, and the model scores a lead **before** that contact. Full audit and evidence: [data_dictionary.md](data_dictionary.md) section 2, [ADR-001](decisions.md).

| Column | Why it leaks |
|---|---|
| `Tags` | Sales status ("Closed by Horizzon", "Ringing"); adds +0.148 PR-AUC on its own |
| `Lead Quality` | Rep's judgment of the lead |
| `Last Activity`, `Last Notable Activity` | Activity at export time, including sales actions (`SMS Sent`) |
| `Lead Profile`, 4 × `Asymmetrique *` | Labels and scores assigned to the lead, timing unknown |

## 5. Columns with no information

12 columns are dropped because they are constant or almost constant (e.g. `Magazine` is 100% `"No"`, `Search` has 14 `"Yes"` in 9,240). See [data_quality.md](data_quality.md) section 2. `Prospect ID` and `Lead Number` are IDs, so they are dropped too.

## 6. Formats and types

| Issue | Example | Fix |
|---|---|---|
| Extra spaces | `"Interested  in full time MBA"` | Strip and collapse spaces (all text columns) |
| Spelling variants | `google` (5) vs. `Google` (2,868) | Merge with `config.json → data.category_aliases` |
| Yes/No as text | `Do Not Email` | `1`/`0` (`Int8`) |
| Counts stored as decimals | `TotalVisits` was `float` only because of NaN | `int` after filling |
| Many tiny groups | `Country`: India 70%, 35 other countries each < 1% | India / Other / Missing |

Other rare categories (e.g. small `Lead Source` values like `bing`, `testone`) are **not** grouped here: that grouping depends on counts, so it is learned on the training split only in the feature pipeline (step 2.1).

## 7. Logic errors

| Check | Rows | Decision |
|---|---|---|
| Target not 0/1 or missing | 0 | `validate` raises an error |
| Negative visits, time or page views | 0 | `validate` raises an error |
| 0 visits but page views > 0 | 0 | `validate` raises an error |
| 0 visits but time on site > 0 | 0 | — |
| Page views per visit between 0 and 1 (impossible) | 0 | — |
| Visits missing but time on site > 0 | 137 | Fill visits (section 1) |
| Visits > 0 but time on site = 0 | 4 | **Keep**: a visit can be shorter than the tracker records |

Checks that can happen in new data are enforced in `validate`, so a broken lead fails loudly instead of being scored.

---

## Final columns

| Column | Type | Values |
|---|---|---|
| `Lead Origin` | text | 5 values |
| `Lead Source` | text | 20 values + `Missing` |
| `Do Not Email` | 0/1 | |
| `TotalVisits` | int | 0–30 |
| `Total Time Spent on Website` | int | 0–2,272 |
| `Page Views Per Visit` | float | 0–15 |
| `Country` | text | India / Other / Missing |
| `Specialization` | text | 18 values + `Missing` |
| `What is your current occupation` | text | 6 values + `Missing` |
| `What matters most to you in choosing a course` | text | 3 values + `Missing` |
| `City` | text | 6 values + `Missing` |
| `A free copy of Mastering The Interview` | 0/1 | |
| `Converted` | 0/1 | target |

## Trade-off

The fill values (3, 2) and caps (30, 15) were read once from the full dataset, test rows included, and then fixed in config. This leaks a tiny amount of information: a median and a round cap from 9,240 rows. In exchange, cleaning is one simple, stateless function that behaves the same in training and in the API. The decision is recorded as [ADR-002](decisions.md).
