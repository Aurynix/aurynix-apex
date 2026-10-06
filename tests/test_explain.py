import numpy as np
import pandas as pd
import pytest

from apex.models.explain import Explainer, _encode
from apex.models.train import make_model


@pytest.fixture
def fitted():
    rng = np.random.default_rng(0)
    n = 300
    visits = rng.integers(0, 6, n).astype(float)
    time = np.where(visits > 0, rng.integers(10, 2000, n), 0)
    occupation = rng.choice(["Working Professional", "Unemployed", None], n)
    leads = pd.DataFrame(
        {
            "Lead Origin": rng.choice(["API", "Landing Page Submission"], n),
            "Do Not Email": rng.choice(["Yes", "No"], n),
            "TotalVisits": visits,
            "Total Time Spent on Website": time,
            "What is your current occupation": occupation,
        }
    )
    y = ((time > 900) | (occupation == "Working Professional")).astype(int)
    model = make_model().fit(leads, y)
    return model, leads, Explainer.fit(model, leads)


def test_contributions_match_shap_linear_explainer(fitted):
    shap = pytest.importorskip("shap")
    model, leads, explainer = fitted
    encoded = _encode(model, leads)
    background = shap.maskers.Independent(encoded, max_samples=len(encoded))  # all rows
    reference = shap.LinearExplainer(model[-1], background).shap_values(encoded[:20])
    assert explainer.contributions(leads.iloc[:20]).sum(axis=1).to_numpy() == pytest.approx(
        reference.sum(axis=1)
    )


def test_contributions_add_up_to_the_score(fitted):
    model, leads, explainer = fitted
    lead = leads.iloc[[0]]
    base = model[-1].intercept_[0] + explainer.means @ model[-1].coef_[0]
    log_odds = base + explainer.contributions(lead).sum(axis=1).iloc[0]
    assert 1 / (1 + np.exp(-log_odds)) == pytest.approx(model.predict_proba(lead)[0, 1])


def test_one_hot_and_web_features_are_grouped_into_readable_reasons(fitted):
    _, leads, explainer = fitted
    fields = set(explainer.contributions(leads.iloc[:5]).columns)
    assert "Website activity" in fields
    assert "What is your current occupation" in fields
    assert not any("_" in f and f.startswith("What is") for f in fields)


def test_explain_one_returns_sorted_reasons(fitted):
    _, leads, explainer = fitted
    result = explainer.explain_one(leads.iloc[[0]])
    assert 0 <= result["probability"] <= 1
    ups = [r["impact"] for r in result["reasons_up"]]
    downs = [r["impact"] for r in result["reasons_down"]]
    assert all(v > 0 for v in ups) and ups == sorted(ups, reverse=True)
    assert all(v < 0 for v in downs) and downs == sorted(downs)
    assert {"feature", "value", "impact"} <= set((result["reasons_up"] + result["reasons_down"])[0])


def test_groups_and_labels_come_from_config(fitted, monkeypatch):
    from apex.models import explain

    _, leads, explainer = fitted
    monkeypatch.setattr(explain, "groups", lambda: {"Profile": ["Lead Origin", "Do Not Email"]})
    monkeypatch.setattr(explain, "labels", lambda: {"Lead Origin": "Origin"})
    fields = set(explainer.contributions(leads.iloc[:5]).columns)
    assert "Profile" in fields
    assert not {"Lead Origin", "Do Not Email"} & fields
