# Architecture Decision Records

Short records of decisions that shape the model. Newest last.

---

## ADR-001: Drop post-contact columns (leakage)

- **Date:** 2026-10-05 · **Step:** 1.4 · **Status:** Accepted

**Context.** Apex scores a lead when it is created, before any sales contact ([problem_framing.md](problem_framing.md)). The dataset is a snapshot taken later, so some columns describe what happened *after* the contact. A model trained on them looks great offline and fails in production.

**Decision.** Drop 9 columns via `config.json → data.leakage_columns`:
`Tags`, `Lead Quality`, `Last Activity`, `Last Notable Activity`, `Lead Profile`, and the 4 `Asymmetrique *` columns.

**Why.**
- `Tags`, `Lead Quality`: the publisher's dictionary says they are the lead's current status and the employee's judgment. A quick model gains +0.148 and +0.061 PR-AUC from them.
- `Last Activity`, `Last Notable Activity`: state at export time, including sales actions such as `SMS Sent`.
- `Lead Profile`, `Asymmetrique *`: assigned labels and scores with unknown timing. Dropped to be safe: the cost is a little signal (≤ +0.022 PR-AUC), the benefit is honest offline numbers.

**Consequences.** Base PR-AUC of a quick model drops from 0.95 (with `Tags`) to 0.805, which is the honest number to beat. Kept but watched: `Lead Origin = Lead Add Form`, website activity, occupation. Details and per-column audit: [data_dictionary.md](data_dictionary.md) section 2.

---

## ADR-002: Fill missing values and cap outliers with fixed values in cleaning

- **Date:** 2026-10-05 · **Step:** data cleaning · **Status:** Accepted

**Context.** Missing values and outliers can be handled in two places: in `clean.py` with fixed rules, or in the model pipeline with values learned on the training split only. The second is stricter but adds moving parts.

**Decision.** Do it in `clean.py` with values fixed in `config.json → data`: text missing → `"Missing"`; `TotalVisits` / `Page Views Per Visit` missing → 3 / 2 (medians); caps 30 / 15; `Country` → India / Other. Count-based rare grouping (e.g. small `Lead Source` values) stays in the model pipeline (step 2.1).

**Why.** Keep it simple: one stateless function cleans the training file and a single API lead the same way, and the cleaned data has no missing values. The leak is tiny: two medians and two round caps read once from 9,240 rows.

**Consequences.** Changing a fill value or cap is a config change and needs a retrain. Details: [data_cleaning.md](data_cleaning.md).
