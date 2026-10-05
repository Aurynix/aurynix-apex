# Data Quality Report

> Stage 3, part 1 (step 1.3). Data: `data/raw/Leads.csv`, 9,240 rows × 37 columns, SHA-256 `1426dffd…3762802` (see [data_dictionary.md](data_dictionary.md)).
> Reproduce with [`notebooks/01_eda.ipynb`](../notebooks/01_eda.ipynb). Cleaning code: [`src/apex/data/clean.py`](../src/apex/data/clean.py).

Every issue found below has a **decision** with a reason. The cleaning rules live in `config.json → data` and are **stateless**: nothing is learned from the data, so the same rules apply to one new lead at prediction time. Filling missing values and capping outliers were added later with fixed values; see [data_cleaning.md](data_cleaning.md) for the full cleaning steps.

---

## Summary

| # | Issue | Size | Decision | Where |
|---|---|---|---|---|
| 1 | `"Select"` = hidden missing value | 4 columns, up to 54.6% of rows | Convert to missing | `clean.py` |
| 2 | `"unknown"` in `Country` | 5 rows | Convert to missing | `clean.py` |
| 3 | High missing rates | 8 columns ≥ 30% | **Keep rows**; missingness is informative. Fill with `"Missing"`; drop `How did you hear…` (78.5%) | `clean.py` ([data_cleaning.md](data_cleaning.md)) |
| 4 | Constant columns | 5 columns | Drop | `clean.py` |
| 5 | Near-constant columns (≤ 14 "Yes" of 9,240) | 7 columns | Drop | `clean.py` |
| 6 | Spelling variants (`google` vs `Google`) | 5 rows | Merge | `clean.py` |
| 7 | Inconsistent whitespace in labels | e.g. `Tags` | Strip and collapse spaces | `clean.py` |
| 8 | Yes/No text columns | 2 kept columns | Encode as 1/0 | `clean.py` |
| 9 | ID columns | 2 columns | Drop from features | `clean.py` |
| 10 | Outliers in `TotalVisits`, `Page Views Per Visit` | max 251 and 55 visits | **Keep rows**; cap at 30 and 15 | `clean.py` ([data_cleaning.md](data_cleaning.md)) |
| 11 | Rows identical except for IDs | 1,489 rows in 193 groups | **Keep**; they are different leads | — |
| 12 | Rare categories | e.g. 21 `Lead Source` values, many with < 10 rows | `Country` → India / Other in `clean.py`; others grouped in step 2.1 (fitted on train) | `clean.py`, step 2.1 |
| 13 | Leakage suspects | see below | 9 columns dropped (step 1.4, ADR-001) | `clean.py` |

**Result:** `data/interim/leads_clean.parquet`, 9,240 rows × 23 columns (no rows removed); 13 columns and no missing values after the leakage audit (step 1.4) and the cleaning in [data_cleaning.md](data_cleaning.md).

---

## 1. Missing values, including hidden ones

`"Select"` is the default option of a form drop-down: the lead did not choose anything. Counting only real `NaN` values **seriously underestimates** missingness:

| Column | NaN | `"Select"` | **Total missing** | Conversion when missing | Conversion when present |
|---|---|---|---|---|---|
| How did you hear about X Education | 23.9% | 54.6% | **78.5%** | 37.4% | 42.6% |
| Lead Profile | 29.3% | 44.9% | **74.2%** | 30.1% | 62.9% |
| Lead Quality | 51.6% | — | **51.6%** | 21.5% | 56.7% |
| Asymmetrique Activity/Profile Index & Score (4 cols) | 45.6% | — | **45.6%** | 39.2% | 38.0% |
| City | 15.4% | 24.3% | **39.7%** | 34.3% | 41.4% |
| Specialization | 15.6% | 21.0% | **36.6%** | 28.7% | 44.2% |
| Tags | 36.3% | — | **36.3%** | 24.9% | 46.3% |
| What matters most to you in choosing a course | 29.3% | — | **29.3%** | 13.7% | 48.9% |
| What is your current occupation | 29.1% | — | **29.1%** | 13.8% | 48.7% |
| Country | 26.6% | — (+5 `"unknown"`) | **26.6%** | 43.7% | 36.7% |
| TotalVisits, Page Views Per Visit | 1.5% | — | **1.5%** | 73.0% | 38.0% |
| Last Activity | 1.1% | — | **1.1%** | | |
| Lead Source | 0.4% | — | **0.4%** | | |

Overall conversion rate: 38.5%.

**Decisions**
- `"Select"` and `"unknown"` → missing (`config.json → data.missing_placeholders`).
- **No rows are dropped and no columns are dropped for missingness alone.** Missingness is strongly linked to the target (e.g. occupation missing: 13.8% conversion vs. 48.7%), so "missing" is signal. `clean.py` fills categoricals with an explicit `"Missing"` category ([data_cleaning.md](data_cleaning.md)).
- ⚠️ **Why** a field is missing matters for leakage: if occupation or "what matters most" are filled in **by the sales rep during the call**, then "present" means "the lead was contacted", which leaks the outcome. This is checked in step 1.4.
- `TotalVisits` and `Page Views Per Visit` are missing on the **same 137 rows**, mostly `Lead Add Form` (110) and `Lead Import` (24) leads, which never visited the website through tracked pages. Their 73% conversion rate is driven by `Lead Add Form` (see section 6).

## 2. Constant and near-constant columns

| Column | Values | Decision |
|---|---|---|
| Magazine · Receive More Updates About Our Courses · Update me on Supply Chain Content · Get updates on DM Content · I agree to pay the amount through cheque | 100% `"No"` | Drop: no information |
| Search (14 Yes) · Through Recommendations (7) · Digital Advertisement (4) · Do Not Call (2) · Newspaper Article (2) · X Education Forums (1) · Newspaper (1) | ≥ 99.85% `"No"` | Drop: too few positives to learn from; one-hot columns that would be all zero in most CV folds |

Listed in `config.json → data.drop_columns` (a fixed list, not recomputed per run, so training and serving always drop the same columns).

## 3. Label consistency

- `Lead Source`: `google` (5) and `Google` (2,868) → merged via `config.json → data.category_aliases`.
- `Tags`: `"Interested  in full time MBA"` has a double space → whitespace is stripped and collapsed for all text columns.
- Other lowercase sources (`bing`, `blog`, `youtubechannel`, `testone`, …) have no capitalized twin, so they stay as-is and are grouped as rare categories in step 2.1.
- `Asymmetrique * Index` values are ordinal strings (`01.High`, `02.Medium`, `03.Low`). Encoding is deferred: these columns are leakage suspects (section 6).

## 4. Binary columns

After dropping constant/near-constant columns, two Yes/No columns remain: `Do Not Email` and `A free copy of Mastering The Interview`. They are encoded as 1/0. Any value other than Yes/No raises an error, so a schema change in incoming data is caught immediately.

## 5. Numeric columns and outliers

| Column | Median | 99th pct | Max | Zeros | Points > Q3 + 1.5·IQR |
|---|---|---|---|---|---|
| TotalVisits | 3 | 17 | **251** | 2,189 | 267 |
| Total Time Spent on Website (seconds) | 248 | 1,841 | 2,272 | 2,193 | 0 |
| Page Views Per Visit | 2 | 9 | **55** | 2,189 | 360 |

- **Consistency checks pass:** no negative values; `TotalVisits` is always a whole number; zero visits always comes with zero time and zero page views. Only 4 leads have visits > 0 but time = 0, which is plausible (bounce).
- **Zeros are real:** the 2,189 zero-activity leads are `API` (1,602), `Lead Add Form` (557) and `Lead Import` (30) leads that never came through the website. Zero is a meaningful value, not a missing one.
- **Decision:** keep all rows. The extreme values (251 visits, 55 pages per visit) are plausible bots or heavy users, not data entry errors. `clean.py` caps them at fixed limits (30 visits, 15 pages per visit; see [data_cleaning.md](data_cleaning.md)). Tree models are barely affected; the cap mainly helps Logistic Regression.
- `validate()` rejects negative numbers and a non-0/1 target, so impossible values fail loudly.

## 6. Duplicates

- **No duplicate IDs** (`Prospect ID`, `Lead Number`) and **no fully duplicated rows**.
- **1,489 rows (193 groups) are identical once IDs are removed.** These are mostly zero-activity API / form leads with all optional fields empty: different people who look the same in this data. 15 groups even have **different outcomes**, which confirms they are separate leads.
- **Decision:** keep them. Removing them would distort the real mix of incoming leads. Splits stay random-stratified; this is noted as a small risk of near-identical rows landing in both train and test, which inflates scores only marginally for such low-information rows.
- IDs are dropped from the feature set (`config.json → data.id_columns`). `Lead Number` looks sequential; if it encodes arrival order it could be useful for a time-based split, but there is no documentation that it does, so it is not used.

## 7. Flags for the leakage audit (step 1.4)

Found during quality checks. Not decided here.

| Column / value | Observation | Concern |
|---|---|---|
| `Lead Origin = Lead Add Form` | 718 leads, **92.5%** conversion (vs. 38.5% overall) | May be leads added by sales after a positive conversation |
| `Tags` | Values like *"Closed by Horizzon"*, *"Ringing"*, *"switched off"*, *"Lost to EINS"* | Clearly written by sales after contact |
| `Lead Quality` | *"High in Relevance"*, *"Worst"* | Sales rep judgment |
| `Last Activity` / `Last Notable Activity` | Values like *"Had a Phone Conversation"*, *"SMS Sent"*, *"Unreachable"* | Post-contact activity |
| `Asymmetrique *` (4 columns) | Proprietary activity/profile scores | Unknown when computed; may include post-contact data |
| Occupation, "What matters most", `Lead Profile` | Presence strongly linked to conversion | May be filled in by sales during the call |

---

## Cleaning rules (config.json → data)

| Key | Value |
|---|---|
| `missing_placeholders` | `["Select", "unknown"]` |
| `category_aliases` | `Lead Source: google → Google` |
| `binary_columns` | `Do Not Email`, `A free copy of Mastering The Interview` |
| `numeric_columns` | `TotalVisits`, `Total Time Spent on Website`, `Page Views Per Visit` (validated ≥ 0) |
| `drop_columns` | 12 constant / near-constant columns (section 2) |
| `id_columns` | `Prospect ID`, `Lead Number` |
| `leakage_columns` | 9 post-contact columns, see [data_dictionary.md](data_dictionary.md) section 2 |
