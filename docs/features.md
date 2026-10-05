# Features

> Stage 5 (step 2.1). Code: [`src/apex/data/features.py`](../src/apex/data/features.py). Ideas and evidence: [eda.md](eda.md).

```bash
make features   # fit the pipeline on all leads and list the 51 features
```

---

## The pipeline

One sklearn `Pipeline` goes from **raw lead rows** (as in `Leads.csv`, without `Converted`) to a **numeric matrix** the model can use:

```
raw lead ─► clean() ─► add_features() ─► scale numbers + one-hot encode text ─► 51 numbers
            stateless   stateless          learned on the training split only
```

| Step | What it does | Learns from data? |
|---|---|---|
| `clean` | All cleaning rules ([data_cleaning.md](data_cleaning.md)) | No |
| `add_features` | Adds 2 features, drops 1 redundant column | No |
| `StandardScaler` | Numbers → mean 0, standard deviation 1 (helps Logistic Regression) | **Yes**: mean and scale |
| `OneHotEncoder` | Each text value → its own 0/1 column | **Yes**: which values are frequent |

Because the whole path is one object, the API can pass a raw lead straight in, and training and serving can never transform data differently. The learned steps must be **fitted on the training split only** (step 2.2), so validation and test stay unseen.

## New features

| Feature | Formula | Why (from [eda.md](eda.md)) |
|---|---|---|
| `time_per_visit` | time on site ÷ visits; 0 when there are no visits | Visit counts alone are flat (29%–43%), time is strong; this separates long, focused visits from quick ones |
| `has_web_activity` | 1 if visits > 0, else 0 | Leads with no web activity are a different group (API and Lead Add Form) |

## Removed

| Column | Why |
|---|---|
| `What matters most to you in choosing a course` | Duplicate: missing for the same leads as occupation (99.3% overlap), and when present it is "Better Career Prospects" in 99.95% of rows |

## Rare and unseen categories

`OneHotEncoder(min_frequency=0.01, handle_unknown="infrequent_if_exist")`:
- Values seen in **less than 1%** of training leads (`config.json → data.rare_category_min_share`) share one column, `<column>_infrequent_sklearn`. Examples: `Lead Source` = `bing`, `Click2call`, `Facebook`.
- A value **never seen** in training (e.g. a new `Lead Source` such as `TikTok`) goes into the same column, so a new lead never breaks the model.

## Quick check

5-fold CV, Logistic Regression, PR-AUC on all 9,240 leads (a sanity check only; the real comparison is step 2.3):

| Version | PR-AUC |
|---|---|
| Clean only | 0.810 ± 0.008 |
| + new features, − `What matters most…` (**chosen**) | **0.816 ± 0.009** |
| + new features, keep `What matters most…` | 0.817 ± 0.009 |

The new features add a little. Dropping `What matters most…` changes nothing beyond noise, so the simpler version is kept.

## Final features (51)

| Group | Columns |
|---|---|
| Numeric (7) | `TotalVisits`, `Total Time Spent on Website`, `Page Views Per Visit`, `time_per_visit`, `has_web_activity`, `Do Not Email`, `A free copy of Mastering The Interview` |
| `Lead Origin` (4) | API, Landing Page Submission, Lead Add Form, infrequent |
| `Lead Source` (8) | Direct Traffic, Google, Olark Chat, Organic Search, Reference, Referral Sites, Welingak Website, infrequent |
| `Country` (3) | India, Other, Missing |
| `Specialization` (17) | 15 specializations, Missing, infrequent |
| `What is your current occupation` (5) | Unemployed, Working Professional, Student, Missing, infrequent |
| `City` (7) | 6 cities, Missing, infrequent (with `Tier II Cities`) |

## Left for later

From [eda.md](eda.md), tested in steps 2.3–2.4 with PR-AUC before any change: drop `Page Views Per Visit`; `Specialization` → present / missing; drop `City`, `Country`, the free-book flag.
