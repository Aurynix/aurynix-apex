import numpy as np
import pandas as pd

from apex.models.explain import Explainer
from apex.models.predict import predict
from apex.models.train import make_model


def test_predict_returns_probability_segment_and_reasons():
    rng = np.random.default_rng(0)
    n = 200
    time = rng.integers(0, 2000, n)
    leads = pd.DataFrame(
        {
            "Lead Origin": rng.choice(["API", "Landing Page Submission"], n),
            "TotalVisits": np.where(time > 0, 3.0, 0.0),
            "Total Time Spent on Website": time,
        }
    )
    model = make_model().fit(leads, (time > 1000).astype(int))
    meta = {
        "segments": {"thresholds": {"high": 0.8, "medium": 0.3}},
        "explainer_means": dict(enumerate(Explainer.fit(model, leads).means)),
    }

    out = predict(leads, model, meta)
    assert list(out.columns) == ["probability", "segment", "top_reason_up", "top_reason_down"]
    assert out["probability"].between(0, 1).all()
    assert set(out["segment"]) <= {"High", "Medium", "Low"}
    assert out.loc[out["probability"] >= 0.8, "segment"].eq("High").all()
    assert out.loc[time.argmax(), "top_reason_up"] == "Website activity"
