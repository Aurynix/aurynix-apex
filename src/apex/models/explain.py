"""Global and per-lead explanations (step 3.4).

For Logistic Regression, SHAP values have an exact closed form:

    contribution_j = weight_j × (value_j − average value_j)

in log-odds, on the model's input features (scaled numbers, one-hot columns).
This is what `shap.LinearExplainer` computes; tests check that both agree.
One-hot columns of the same raw field are summed, so a reason reads
"Occupation = Working Professional", not 5 separate columns. The four website
features are summed into one "Website activity" reason: they work together
(no visit converts better than a short visit), so showing them separately
would give a rep contradicting reasons.

Run `python -m apex.models.explain` (or `make explain`) for the global
ranking, a figure, and a few example leads.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

WEB = ["Total Time Spent on Website", "TotalVisits", "time_per_visit", "has_web_activity"]
LABELS = {
    "Lead Origin": "Lead origin",
    "Lead Source": "Lead source",
    "Do Not Email": "Opted out of email",
    "Specialization": "Specialization",
    "What is your current occupation": "Occupation",
}


@dataclass
class Explainer:
    """Explains a fitted `make_model()` pipeline against a background of typical leads."""

    model: Pipeline
    means: np.ndarray  # average of each model input over the background leads

    @classmethod
    def fit(cls, model: Pipeline, background: pd.DataFrame) -> "Explainer":
        return cls(model, _encode(model, background).mean(axis=0))

    def contributions(self, leads: pd.DataFrame) -> pd.DataFrame:
        """Log-odds contribution of each raw field, one row per lead."""
        weights = self.model[-1].coef_[0]
        per_column = (_encode(self.model, leads) - self.means) * weights
        columns = pd.DataFrame(per_column, columns=self._input_names(), index=leads.index)
        return columns.T.groupby(self._source_fields()).sum().T

    def explain(self, leads: pd.DataFrame, top: int = 3) -> list[dict]:
        """Top reasons up and down for each lead (one dict per row)."""
        contributions = self.contributions(leads)
        values = self.model[0][:-1].transform(leads)  # as the model sees them
        results = []
        for (_, row), (_, lead_values) in zip(
            contributions.iterrows(), values.iterrows(), strict=True
        ):
            reasons = [
                {
                    "feature": LABELS.get(f, f),
                    "value": _readable(lead_values, f),
                    "impact": round(float(v), 3),
                }
                for f, v in row.sort_values(key=abs, ascending=False).items()
                if v != 0
            ]
            results.append(
                {
                    "reasons_up": [r for r in reasons if r["impact"] > 0][:top],
                    "reasons_down": [r for r in reasons if r["impact"] < 0][:top],
                }
            )
        return results

    def explain_one(self, lead: pd.DataFrame, top: int = 3) -> dict:
        """Score plus the top reasons up and down for one lead (a one-row DataFrame)."""
        probability = round(float(self.model.predict_proba(lead)[0, 1]), 3)
        return {"probability": probability, **self.explain(lead, top)[0]}

    def importance(self, leads: pd.DataFrame) -> pd.Series:
        """Global importance: mean absolute contribution per raw field, largest first."""
        mean_abs = self.contributions(leads).abs().mean()
        return mean_abs.rename(index=lambda f: LABELS.get(f, f)).sort_values(ascending=False)

    def _input_names(self) -> list[str]:
        return list(self.model[0][-1].get_feature_names_out())

    def _source_fields(self) -> list[str]:
        """The raw field behind each model input (one-hot columns map to their field)."""
        categorical = self.model[0][-1].transformers_[1][2]
        fields = []
        for name in self._input_names():
            match = [c for c in categorical if name.startswith(f"{c}_")]
            fields.append(match[0] if match else "Website activity" if name in WEB else name)
        return fields


def _encode(model: Pipeline, leads: pd.DataFrame) -> np.ndarray:
    """Raw leads → the exact inputs of the Logistic Regression."""
    return np.asarray(model[0].transform(leads), dtype=float)


def _readable(values: pd.Series, field: str) -> str:
    """The lead's value of a field, as a short text."""
    if field == "Website activity":
        visits, minutes = values["TotalVisits"], values["Total Time Spent on Website"] / 60
        return f"{visits:.0f} visit{'' if visits == 1 else 's'}, {minutes:.0f} min on site"
    if field == "Do Not Email":
        return "yes" if values[field] == 1 else "no"
    return str(values[field])


def run() -> pd.Series:
    """Global ranking, figure, and three example leads on the validation split."""
    from apex.config import load_config, path
    from apex.data.split import load_splits
    from apex.models.train import make_model

    cfg = load_config()
    target = cfg["data"]["target"]
    parts = load_splits()
    X_train, y_train = parts["train"].drop(columns=[target]), parts["train"][target]
    X_val = parts["val"].drop(columns=[target])

    model = make_model(cfg).fit(X_train, y_train)
    explainer = Explainer.fit(model, X_train)
    ranking = explainer.importance(X_val)

    print("Global importance (mean |log-odds contribution|, validation):\n")
    print(ranking.round(3).to_string(), "\n")
    figure = path("figures_dir") / "explain_importance.png"
    _plot_importance(ranking, figure)

    scores = model.predict_proba(X_val)[:, 1]
    for label, i in [("highest", scores.argmax()), ("middle", np.argsort(scores)[len(scores) // 2]),
                     ("lowest", scores.argmin())]:  # fmt: skip
        result = explainer.explain_one(X_val.iloc[[i]])
        print(f"{label} score: {result['probability']}")
        for r in result["reasons_up"] + result["reasons_down"]:
            print(f"   {r['impact']:+.2f}  {r['feature']} = {r['value']}")
    print(f"\nfigure: {figure}")
    return ranking


def _plot_importance(ranking: pd.Series, file) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, surface = "#0b0b0b", "#52514e", "#fcfcfb"
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=surface)
    ax.barh(ranking.index[::-1], ranking.values[::-1], color="#2a78d6", height=0.6)
    for y, v in enumerate(ranking.values[::-1]):
        ax.text(v + 0.01, y, f"{v:.2f}", va="center", fontsize=8, color=muted)
    ax.set_title(
        "What drives the score (mean |contribution|, log-odds)",
        loc="left",
        fontsize=10,
        color=ink,
        fontweight="bold",
    )
    ax.set_facecolor(surface)
    ax.tick_params(colors=muted, labelsize=8, length=0)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_xticks([])
    fig.tight_layout()
    fig.savefig(file, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    run()
