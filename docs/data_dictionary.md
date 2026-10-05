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

> Step 1.4. Meanings come from the publisher's `Leads Data Dictionary.xlsx`. Missing % counts `"Select"` as missing.

The **prediction-moment test** for every column: *is this known when the lead enters the SDR queue, before any sales contact?* (see [problem_framing.md](problem_framing.md)). Status: ✅ yes · ❌ no · ❓ uncertain.

| Column | Type | Meaning | Missing % | Available at prediction time? | Decision | Reason |
|---|---|---|---|---|---|---|
| Prospect ID | id | Unique lead ID | 0 | ✅ | Drop | ID, no signal (step 1.3) |
| Lead Number | id | Number assigned to each lead | 0 | ✅ | Drop | ID, no signal (step 1.3) |
| Lead Origin | category | How the lead was created (API, Landing Page Submission, Lead Add Form, …) | 0 | ✅ | Keep | Set when the lead is created. ⚠️ `Lead Add Form` converts at 92.5%; watched as a risk |
| Lead Source | category | Where the lead came from (Google, Olark Chat, …) | 0.4 | ✅ | Keep | Set when the lead is created |
| Do Not Email | binary | Lead opted out of emails | 0 | ✅ | Keep | Chosen by the lead on the form |
| Do Not Call | binary | Lead opted out of calls | 0 | ✅ | Drop | Near-constant (step 1.3) |
| Converted | target | Lead became a customer | 0 | — | Target | — |
| TotalVisits | numeric | Number of website visits | 1.5 | ❓ | Keep | Website behaviour before contact; may keep growing afterwards (risk below) |
| Total Time Spent on Website | numeric | Total time on the website | 0 | ❓ | Keep | Same as `TotalVisits` |
| Page Views Per Visit | numeric | Average pages per visit | 1.5 | ❓ | Keep | Same as `TotalVisits` |
| Last Activity | category | Last activity on the lead (Email Opened, SMS Sent, Had a Phone Conversation, …) | 1.1 | ❌ | **Leakage** | "Last" = state at data export; includes sales actions (`SMS Sent` → 69% conversion) |
| Country | category | Country of the lead | 26.6 | ✅ | Keep | Form / IP data |
| Specialization | category | Industry the lead worked in | 36.6 | ✅ | Keep | Selected by the lead on the form |
| How did you hear about X Education | category | Where the lead heard of X Education | 78.5 | ✅ | Drop | Selected by the lead, but 78.5% missing and weak signal ([data_cleaning.md](data_cleaning.md)) |
| What is your current occupation | category | Student, unemployed, working professional, … | 29.1 | ✅ | Keep | Selected by the lead ("option selected by the customer") |
| What matters most to you in choosing a course | category | Main reason for taking a course | 29.3 | ✅ | Keep | Selected by the lead |
| Search, Magazine, Newspaper Article, X Education Forums, Newspaper, Digital Advertisement, Through Recommendations | binary | Where the lead saw an ad | 0 | ✅ | Drop | Constant / near-constant (step 1.3) |
| Receive More Updates About Our Courses, Update me on Supply Chain Content, Get updates on DM Content, I agree to pay the amount through cheque | binary | Lead preferences | 0 | ✅ | Drop | Constant (step 1.3) |
| Tags | category | "Current status of the lead" (Closed by Horizzon, Ringing, Lost to EINS, …) | 36.3 | ❌ | **Leakage** | Written by sales after contact; +0.148 PR-AUC |
| Lead Quality | category | "Based on the data and intuition of the employee assigned to the lead" | 51.6 | ❌ | **Leakage** | Sales rep judgment after assignment; +0.061 PR-AUC |
| Lead Profile | category | "A lead level assigned to each customer" (Potential Lead, Student of SomeSchool, …) | 74.2 | ❓ | **Leakage** | Assigned label, not chosen by the lead; `Potential Lead` → 79% conversion. Dropped to be safe |
| City | category | City of the lead | 39.7 | ✅ | Keep | Selected by the lead on the form |
| Asymmetrique Activity Index / Profile Index / Activity Score / Profile Score | category / numeric | Proprietary index and score "based on activity and profile" | 45.6 | ❓ | **Leakage** | Unknown when computed; "activity" may include post-contact activity. Dropped to be safe |
| A free copy of Mastering The Interview | binary | Lead asked for the free book | 0 | ✅ | Keep | Chosen by the lead on the form |
| Last Notable Activity | category | Last notable activity on the lead | 0 | ❌ | **Leakage** | Same as `Last Activity` |

**Result:** 9 leakage columns in `config.json → data.leakage_columns`, dropped by `clean.py`. After all cleaning the data has **9,240 rows × 13 columns** (12 features + target); see [data_cleaning.md](data_cleaning.md).

### Evidence: quick model with and without suspects

`make leakage` (`src/apex/data/leakage.py`): logistic regression, 5-fold CV, PR-AUC. Base = cleaned data without any suspect; each row adds one group. Random guessing scores 0.385 (the conversion rate).

| Columns | PR-AUC | Gain |
|---|---|---|
| base (no suspects) | 0.805 | — |
| + Tags | **0.952** | **+0.148** |
| + Lead Quality | 0.866 | +0.061 |
| + Last Activity, Last Notable Activity | 0.841 | +0.036 |
| + Asymmetrique (4 columns) | 0.827 | +0.022 |
| + Lead Profile | 0.824 | +0.020 |

`Tags` alone lifts PR-AUC by 0.15: a model with it would look excellent offline and fail in production, where tags do not exist yet. The smaller gains are not proof of leakage on their own; those columns are dropped because of **what they mean** (see table).

### Remaining risks

- **Lead Add Form** (718 leads, 92.5% conversion): may be leads entered by sales after a good conversation. Kept, because origin is known at creation; checked again in error analysis.
- **Website activity** (`TotalVisits`, time on site, page views): the export date is unknown, so counts may include visits after first contact. Kept, because they are the main pre-contact behaviour signal.
- **Occupation / what matters most**: the publisher says the lead selects them, but presence is strongly linked to conversion (step 1.3). Kept, and watched in SHAP (step 3.4).
