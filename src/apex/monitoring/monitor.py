"""Compare recent predictions with the reference profile (step 4.4).

Three checks on the leads scored in the last `window_days` (config.json → monitoring):

1. Feature drift: PSI of each model input (time on site, visits, occupation, ...)
   against the training distribution, with the bin edges fixed at training time.
2. Prediction drift: PSI of the score, and High / Medium / Low shares against
   training (20 / 30 / 50).
3. Data quality: missing rate per field, share of categories never seen in
   training, share of leads that relied on defaults, invalid requests rejected
   by the API.

Fewer than `min_samples` predictions → status `insufficient_data`. Each run is
saved in SQLite (`drift_runs`) and as reports/monitoring/latest.json.

Run `python -m apex.monitoring.monitor` (or `make monitor`).
"""

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from apex.api.database import (
    count_rejections,
    drift_runs,
    now,
    read_predictions,
    save_drift_run,
)
from apex.config import load_config, path
from apex.monitoring.drift import bin_shares, category_shares, psi, status, worst


class Monitor:
    def __init__(
        self,
        model: Pipeline,
        profile: dict[str, Any],
        db: sqlite3.Connection,
        config: dict[str, Any] | None = None,
    ):
        self.model, self.profile, self.db = model, profile, db
        self.cfg = config or load_config()

    def run(self, window_days: int | None = None, save: bool = True) -> dict[str, Any]:
        """Check the last `window_days` of predictions; save and return the report."""
        mon = self.cfg["monitoring"]
        days = window_days or mon["window_days"]
        since = (datetime.now(UTC) - timedelta(days=days)).isoformat(timespec="seconds")
        rows = read_predictions(self.db, since)
        report: dict[str, Any] = {
            "created_at": now(),
            "window_days": days,
            "n_samples": len(rows),
            "model_version": self.profile["model_version"],
        }
        if len(rows) < mon["min_samples"]:
            report["status"] = "insufficient_data"
        else:
            leads = pd.DataFrame([r[0] for r in rows], columns=self.cfg["serving"]["input_fields"])
            report["features"] = self.feature_drift(leads)
            report["prediction"] = self.prediction_drift([r[1] for r in rows], [r[2] for r in rows])
            report["data_quality"] = self.data_quality(leads, report["features"], since)
            report["status"] = worst(
                [f["status"] for f in report["features"].values()]
                + [report["prediction"]["status"], report["data_quality"]["status"]]
            )
        if save:
            report["id"] = save_drift_run(self.db, report)
            out = path("monitoring_reports_dir")
            out.mkdir(parents=True, exist_ok=True)
            (out / "latest.json").write_text(json.dumps(report, indent=2) + "\n")
        return report

    def feature_drift(self, leads: pd.DataFrame) -> dict[str, dict]:
        """PSI per model input against the training distribution."""
        inputs = self.model[0][:-1].transform(leads)  # clean + features, as the model sees them
        other = self.cfg["monitoring"]["unseen_category_label"]
        result = {}
        for col, ref in self.profile["numeric"].items():
            value = psi(ref["shares"], bin_shares(inputs[col], ref["edges"]))
            result[col] = {
                "type": "numeric",
                "psi": round(value, 4),
                "status": status(value, self.cfg),
            }
        for col, ref in self.profile["categorical"].items():
            current = category_shares(inputs[col], list(ref), other)
            value = psi([*ref.values(), 0.0], list(current.values()))
            result[col] = {
                "type": "categorical",
                "psi": round(value, 4),
                "status": status(value, self.cfg),
                "unseen_share": round(current[other], 4),
            }
        return result

    def prediction_drift(self, scores: list[float], segments: list[str]) -> dict[str, Any]:
        """Score PSI and segment shares against training."""
        ref_score = self.profile["score"]
        score_psi = psi(ref_score["shares"], bin_shares(pd.Series(scores), ref_score["edges"]))
        ref_seg = {k.lower(): v for k, v in self.profile["segment_shares"].items()}
        cur_seg = pd.Series(segments).str.lower().value_counts(normalize=True)
        cur_seg = {k: round(float(cur_seg.get(k, 0.0)), 4) for k in ("high", "medium", "low")}
        segment_psi = psi([ref_seg[k] for k in cur_seg], list(cur_seg.values()))
        return {
            "score_psi": round(score_psi, 4),
            "segment_psi": round(segment_psi, 4),
            "mean_score": round(float(pd.Series(scores).mean()), 4),
            "segment_shares": {
                "reference": {k: round(ref_seg[k], 4) for k in cur_seg},
                "current": cur_seg,
            },
            "status": worst([status(score_psi, self.cfg), status(segment_psi, self.cfg)]),
        }

    def data_quality(self, leads: pd.DataFrame, features: dict, since: str) -> dict[str, Any]:
        """Missing fields, unseen categories, defaults, and rejected requests."""
        mon = self.cfg["monitoring"]
        hidden = self.cfg["data"]["missing_placeholders"]
        missing = leads.isna() | leads.isin(hidden)

        missing_rate, statuses = {}, []
        for col, ref in self.profile["missing_rate"].items():
            cur = float(missing[col].mean())
            change = cur - ref
            flag = "warning" if change >= mon["missing_rate_change_warning"] else "ok"
            missing_rate[col] = {
                "reference": round(ref, 4),
                "current": round(cur, 4),
                "status": flag,
            }
            statuses.append(flag)

        unseen = {c: f["unseen_share"] for c, f in features.items() if f["type"] == "categorical"}
        statuses += [
            "warning" if v >= mon["unseen_share_warning"] else "ok" for v in unseen.values()
        ]

        rejected = count_rejections(self.db, since)
        rejection_rate = rejected / (rejected + len(leads))
        statuses.append("warning" if rejection_rate >= mon["rejection_rate_warning"] else "ok")
        return {
            "missing_rate": missing_rate,
            "unseen_category_share": unseen,
            "defaulted_share": round(float(missing.any(axis=1).mean()), 4),
            "rejected_requests": rejected,
            "rejection_rate": round(rejection_rate, 4),
            "status": worst(statuses),
        }

    def history(self, limit: int = 30) -> list[dict[str, Any]]:
        return drift_runs(self.db, limit)


def summary(report: dict[str, Any]) -> str:
    """A short text version of a report, for the terminal."""
    lines = [f"Status: {report['status'].upper()}  ({report['n_samples']:,} predictions, "
             f"last {report['window_days']} days)"]  # fmt: skip
    if "features" not in report:
        return "\n".join(lines)
    pred, dq = report["prediction"], report["data_quality"]
    lines += ["", "Feature drift (PSI):"]
    for name, f in sorted(report["features"].items(), key=lambda kv: -kv[1]["psi"]):
        extra = f", unseen {f['unseen_share']:.0%}" if f.get("unseen_share") else ""
        lines.append(f"  {f['status']:<8} {f['psi']:>7.3f}  {name}{extra}")
    ref, cur = pred["segment_shares"]["reference"], pred["segment_shares"]["current"]
    lines += [
        "",
        f"Prediction drift: {pred['status']}  (score PSI {pred['score_psi']:.3f}, "
        f"segment PSI {pred['segment_psi']:.3f}, mean score {pred['mean_score']:.3f})",
        "  segment   training  now",
        *[f"  {k:<8}  {ref[k]:>7.0%}  {cur[k]:>4.0%}" for k in ("high", "medium", "low")],
        "",
        f"Data quality: {dq['status']}  (leads with defaults {dq['defaulted_share']:.0%}, "
        f"rejected requests {dq['rejected_requests']} = {dq['rejection_rate']:.1%})",
    ]
    for name, m in dq["missing_rate"].items():
        if m["status"] != "ok":
            lines.append(f"  missing {name}: {m['reference']:.0%} → {m['current']:.0%}")
    for name, share in dq["unseen_category_share"].items():
        if share >= 0.01:
            lines.append(f"  unseen categories in {name}: {share:.0%} of leads")
    return "\n".join(lines)


def create_monitor(db: sqlite3.Connection | None = None, model: Pipeline | None = None) -> Monitor:
    """Monitor for the saved model, its reference profile, and the API database."""
    from apex.api.database import connect
    from apex.models.predict import load_model

    cfg = load_config()
    model = model or load_model()[0]
    with (path("models_dir") / cfg["files"]["reference_profile"]).open(encoding="utf-8") as f:
        profile = json.load(f)
    return Monitor(model, profile, db or connect(path("database")), cfg)


if __name__ == "__main__":
    print(summary(create_monitor().run()))
