"""Streamlit demo for Aurynix Apex.

    make run    # the API must be running first
    make demo   # → http://localhost:8501

The demo is a client of the API (`API_URL`, default http://localhost:8000):
it never loads the model itself. Three tabs: score one lead, score a CSV,
and drift monitoring.
"""

import json
import os
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import altair as alt
import pandas as pd
import streamlit as st

from apex.config import load_config

LOGO = str(Path(__file__).with_name("icon.svg"))
DEMO_SOURCE = "demo_seed"
DISPLAY_TIMEZONE = ZoneInfo("Asia/Jerusalem")  # follows daylight saving by itself
DISPLAY_TIMEZONE_NAME = "Jerusalem time"
API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.environ.get("API_KEY", "").strip()  # server side only: never shown to visitors
# How long to wait for a sleeping API to wake up (free hosting). 0 = fail at once (local use).
API_WAKE_SECONDS = float(os.environ.get("API_WAKE_SECONDS", "0"))
WAKE_RETRY_SECONDS = 5
WAKE_STATUS_CODES = (502, 503, 504)  # what a host may answer while the API is starting

# Raw Kaggle column → API field, so Leads.csv can be uploaded as-is.
RAW_TO_API = {
    "Lead Origin": "lead_origin",
    "Lead Source": "lead_source",
    "Do Not Email": "do_not_email",
    "TotalVisits": "total_visits",
    "Total Time Spent on Website": "time_on_website",
    "Specialization": "specialization",
    "What is your current occupation": "occupation",
}
FIELDS = list(RAW_TO_API.values())
SEGMENT_ICON = {"high": "🟢 High", "medium": "🟡 Medium", "low": "🔴 Low"}
STATUS_ICON = {"ok": "🟢 ok", "warning": "🟡 warning", "drift": "🔴 drift"}
SEGMENT_ORDER = ["High", "Medium", "Low"]
# Monitoring field names → the labels used on the Score a lead tab.
FIELD_LABELS = {
    "Lead Origin": "Lead origin",
    "Lead Source": "Lead source",
    "Do Not Email": "Opted out of emails",
    "TotalVisits": "Website visits",
    "Total Time Spent on Website": "Time on website",
    "Specialization": "Specialization",
    "What is your current occupation": "Occupation",
    "time_per_visit": "Time per visit",
    "has_web_activity": "Has web activity",
}
# Score a CSV: input fields shown next to each score (and kept in the download).
BATCH_INPUT_COLUMNS = [
    "lead_origin",
    "lead_source",
    "occupation",
    "total_visits",
    "time_on_website",
]


def field_label(name: str) -> str:
    """A readable label for a monitored field; unknown names are tidied, not hidden."""
    return FIELD_LABELS.get(name, name.replace("_", " ").capitalize())


def psi_help(mon: dict) -> str:
    """Explanation shown above the feature drift table (thresholds from config.json)."""
    warn, drift = mon["psi_warning"], mon["psi_drift"]
    return f"""\
**PSI (Population Stability Index)** measures how much each field has changed \
since training. In other words: are the leads coming in now the same kind of people?

| PSI | Status | Meaning |
|---|---|---|
| below {warn:.2f} | 🟢 ok | Stable: almost the same people |
| {warn:.2f} to {drift:.2f} | 🟡 warning | Noticeable change: the audience is starting to differ, \
investigate |
| {drift:.2f} or more | 🔴 drift | Big change: a different audience |
"""


def data_quality_help(mon: dict) -> str:
    """Explanation shown above the data quality table (threshold from config.json)."""
    points = mon["missing_rate_change_warning"] * 100
    return f"""\
This is not PSI. It shows the **share of missing values** in each field, in training \
(reference) and now (current). A field gets a warning if its missing rate rises by \
{points:g} points or more, which usually means a field stopped being sent.
"""


def segment_chart(shares: dict) -> alt.Chart:
    """Grouped bars of segment shares, High → Medium → Low, y-axis from 0."""
    data = (
        pd.DataFrame(shares)
        .rename(index=str.capitalize)
        .rename_axis("segment")
        .reset_index()
        .melt(id_vars="segment", var_name="period", value_name="share")
    )
    return (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X("segment:N", sort=SEGMENT_ORDER, title=None, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("period:N", sort=["reference", "current"]),
            y=alt.Y(
                "share:Q",
                title="Share of leads",
                scale=alt.Scale(domainMin=0, zero=True, nice=True),
                axis=alt.Axis(format="%"),
            ),
            color=alt.Color("period:N", sort=["reference", "current"], title=None),
            tooltip=["segment", "period", alt.Tooltip("share:Q", format=".1%")],
        )
    )


def ranked_leads(df: pd.DataFrame, leads: list[dict], result: list[dict]) -> pd.DataFrame:
    """Scored leads, best first, with the file row number and the main input fields."""
    ranked = pd.DataFrame(
        {
            "row": range(1, len(df) + 1),  # 1 = the first data row of the uploaded file
            "score": [p["score"] for p in result],
            "segment": [SEGMENT_ICON[p["segment"]] for p in result],
        }
    )
    inputs = pd.DataFrame(leads).reindex(columns=BATCH_INPUT_COLUMNS)
    ranked = pd.concat([ranked, inputs], axis=1)
    ranked["top reason up"] = [", ".join(p["reasons"]["up"][:1]) for p in result]
    ranked["top reason down"] = [", ".join(p["reasons"]["down"][:1]) for p in result]
    if "Prospect ID" in df.columns:
        ranked.insert(1, "Prospect ID", df["Prospect ID"].to_numpy())
    return ranked.sort_values("score", ascending=False, kind="stable")


def ranked_for_screen(ranked: pd.DataFrame) -> pd.DataFrame:
    """The on-screen table: time in minutes, like the Score a lead form. The download keeps
    seconds under the API name, so the file can be uploaded and scored again."""
    shown = ranked.copy()
    shown["time_on_website"] = (pd.to_numeric(shown["time_on_website"]) / 60).round(1)
    return shown.rename(columns={"time_on_website": "Time on website (min)"})


def time_ago(seconds: float) -> str:
    """A duration in words, for example "2 min ago"."""
    minutes = int(seconds // 60)
    if minutes < 1:
        return "just now"
    if minutes < 60:
        return f"{minutes} min ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours} h ago"
    days = hours // 24
    return f"{days} day{'s' if days > 1 else ''} ago"


def last_run(created_at: str, now: datetime | None = None) -> str:
    """An ISO timestamp as "last run 2 min ago (7 Nov 2026, 00:48 Jerusalem time)"."""
    try:
        when = datetime.fromisoformat(created_at).astimezone(DISPLAY_TIMEZONE)
    except ValueError:
        return f"last run {created_at}"
    now = now or datetime.now(UTC)
    shown = f"{when.day} {when:%b %Y, %H:%M} {DISPLAY_TIMEZONE_NAME}"
    return f"last run {time_ago((now - when).total_seconds())} ({shown})"


class ApiUnavailable(RuntimeError):
    """The API cannot be reached right now, or is still starting."""


def call(method: str, path: str, payload: dict | None = None) -> dict:
    """Call the API and return its JSON; raises RuntimeError with a readable message."""
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"content-type": "application/json"}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    request = urllib.request.Request(f"{API_URL}{path}", data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as err:
        if err.code in WAKE_STATUS_CODES:
            raise ApiUnavailable(f"The API at {API_URL} is not ready (HTTP {err.code}).") from err
        raise RuntimeError(f"API error {err.code}: {err.read().decode()[:500]}") from err
    except urllib.error.URLError as err:
        raise ApiUnavailable(f"Cannot reach the API at {API_URL} ({err.reason}).") from err
    except TimeoutError as err:
        raise ApiUnavailable(f"Cannot reach the API at {API_URL} (timed out).") from err


def wait_for_api(path: str, wait_seconds: float) -> dict:
    """GET `path`, retrying while the API is unavailable, for up to `wait_seconds`."""
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            return call("GET", path)
        except ApiUnavailable:
            if time.monotonic() >= deadline:
                raise
            time.sleep(WAKE_RETRY_SECONDS)


def leads_from_csv(df: pd.DataFrame) -> list[dict]:
    """CSV rows → API leads. Accepts API field names or the raw Kaggle columns."""
    df = df.rename(columns=RAW_TO_API)
    if "lead_origin" not in df.columns:
        raise ValueError("The CSV needs a 'lead_origin' (or 'Lead Origin') column.")
    df = df[[c for c in FIELDS if c in df.columns]].copy()
    df = df.replace({"Select": None})
    if "do_not_email" in df.columns:
        df["do_not_email"] = df["do_not_email"].map(
            lambda v: str(v).strip().lower() in ("yes", "true", "1")
        )
    for col in ("total_visits", "time_on_website"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round().astype("Int64")
    leads = df.astype(object).where(df.notna(), None).to_dict(orient="records")
    return [{k: v for k, v in lead.items() if v is not None} for lead in leads]


def show_prediction(result: dict) -> None:
    col1, col2 = st.columns(2)
    col1.metric("Probability to convert", f"{result['score']:.0%}")
    col2.metric("Priority", SEGMENT_ICON[result["segment"]])
    up, down = st.columns(2)
    up.markdown("**Reasons up ↑**\n" + "".join(f"\n- {r}" for r in result["reasons"]["up"]))
    down.markdown("**Reasons down ↓**\n" + "".join(f"\n- {r}" for r in result["reasons"]["down"]))


def tab_single() -> None:
    with st.form("lead"):
        c1, c2 = st.columns(2)
        origin = c1.selectbox(
            "Lead origin",
            ["Landing Page Submission", "API", "Lead Add Form", "Lead Import", "Quick Add Form"],
        )
        source = c2.selectbox(
            "Lead source",
            ["Google", "Direct Traffic", "Olark Chat", "Organic Search", "Reference",
             "Welingak Website", "Referral Sites", "Facebook", "(unknown)"],
        )  # fmt: skip
        visits = c1.number_input("Website visits", min_value=0, value=3)
        minutes = c2.number_input("Time on website (minutes)", min_value=0, value=15)
        occupation = c1.selectbox(
            "Occupation",
            ["Unemployed", "Working Professional", "Student", "Other", "(not given)"],
        )
        specialization = c2.selectbox(
            "Specialization", ["Finance Management", "Marketing Management", "(not given)"]
        )
        no_email = st.checkbox("Opted out of emails")
        submitted = st.form_submit_button("Score lead", type="primary")
    if submitted:
        lead = {
            "lead_origin": origin,
            "lead_source": None if source == "(unknown)" else source,
            "do_not_email": no_email,
            "total_visits": int(visits),
            "time_on_website": int(minutes * 60),
            "occupation": None if occupation == "(not given)" else occupation,
            "specialization": None if specialization == "(not given)" else specialization,
        }
        show_prediction(call("POST", "/predict/single", lead))


@st.cache_data(ttl=600, max_entries=20, show_spinner=False)
def score_batch(leads: list[dict]) -> list[dict]:
    """Score a batch once: Streamlit reruns the page on every click, and each scoring
    would log the same leads again."""
    return call("POST", "/predict/batch", {"leads": leads})["predictions"]


def tab_batch(limit: int) -> None:
    st.caption(
        "Upload a CSV with the API fields (`lead_origin`, `lead_source`, `do_not_email`, "
        "`total_visits`, `time_on_website`, `specialization`, `occupation`) or the raw "
        f"Kaggle columns (e.g. `Leads.csv`). Up to {limit:,} rows."
    )
    file = st.file_uploader("Leads CSV", type="csv")
    if not file:
        return
    df = pd.read_csv(file)
    if len(df) > limit:
        st.info(f"This demo scores the first {limit:,} of {len(df):,} rows.")
        df = df.head(limit)
    leads = leads_from_csv(df)
    result = score_batch(leads)
    ranked = ranked_leads(df, leads, result)
    counts = ranked["segment"].value_counts()
    cols = st.columns(3)
    for col, seg in zip(cols, SEGMENT_ICON.values(), strict=True):
        col.metric(seg, f"{counts.get(seg, 0):,}")
    st.dataframe(ranked_for_screen(ranked), width="stretch", hide_index=True)
    st.download_button("Download ranked leads", ranked.to_csv(index=False), "ranked_leads.csv")


def tab_monitoring() -> None:
    if st.button("Run monitoring now (last 7 days)"):
        call("POST", "/monitoring/run")
    try:
        report = call("GET", "/monitoring/latest")
    except RuntimeError:
        st.info("No monitoring run yet. Score some leads, then run monitoring.")
        return
    st.metric("Status", STATUS_ICON.get(report["status"], report["status"]))
    st.caption(f"{report['n_samples']:,} predictions · {last_run(report['created_at'])}")
    if report.get("data_source"):
        st.caption(
            "This public demo reports on synthetic sample leads only, with recent timestamps, "
            "so leads you score here do not change it."
        )
    if report.get("retrain_recommended"):
        st.error("Retraining recommended.")

    perf = report.get("performance", {})
    st.subheader("Performance on recorded outcomes")
    if perf.get("status", "insufficient_data") == "insufficient_data":
        st.info(
            f"Not enough outcomes yet ({perf.get('n_outcomes', 0):,}). "
            "Send real results to `POST /outcomes` with each lead's `prediction_id`."
        )
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Status", STATUS_ICON.get(perf["status"], perf["status"]))
        act, exp = perf["actual"], perf["expected"]
        is_demo = report.get("data_source") == DEMO_SOURCE
        # A better score than the test result is no achievement on demo data: keep it neutral.
        delta_color = "off" if is_demo else "normal"
        c2.metric(
            "PR-AUC",
            f"{act['pr_auc']:.3f}",
            f"{act['pr_auc'] - exp['pr_auc']:+.3f} vs. test",
            delta_color=delta_color,
        )
        if act["precision_high"] is not None:
            c3.metric(
                "High segment precision",
                f"{act['precision_high']:.0%}",
                f"{act['precision_high'] - exp['precision_high']:+.0%} vs. test",
                delta_color=delta_color,
            )
        st.caption(f"{perf['n_outcomes']:,} outcomes in the last {perf['window_days']} days")
        if is_demo:
            st.caption(
                "Synthetic demo outcomes are drawn from the model's own probabilities, so this "
                f"is higher than the real held-out test result ({exp['pr_auc']:.3f}). "
                "It is not evidence of quality."
            )

    if "features" not in report:
        st.warning(f"Not enough predictions for a reliable check (status: {report['status']}).")
        return

    mon = load_config()["monitoring"]
    shares = report["prediction"]["segment_shares"]
    st.subheader("Segments: training vs. now")
    st.altair_chart(segment_chart(shares), width="stretch")

    st.subheader("Feature drift (PSI)")
    st.markdown(psi_help(mon))
    features = pd.DataFrame(report["features"]).T.sort_values("psi", ascending=False)
    features["status"] = features["status"].map(STATUS_ICON)
    features = features.rename(index=field_label).rename_axis("field")
    st.dataframe(features[["status", "psi"]], width="stretch")

    dq = report["data_quality"]
    st.subheader(f"Data quality: {STATUS_ICON.get(dq['status'], dq['status'])}")
    st.markdown(data_quality_help(mon))
    missing = pd.DataFrame(dq["missing_rate"]).T[["reference", "current", "status"]]
    missing = missing.rename(index=field_label).rename_axis("field")
    for col in ("reference", "current"):
        missing[col] = pd.to_numeric(missing[col]) * 100
    percent = st.column_config.NumberColumn(format="%.1f%%")
    st.dataframe(missing, width="stretch", column_config={"reference": percent, "current": percent})
    st.caption(
        f"Leads relying on defaults: {dq['defaulted_share']:.0%} · "
        f"rejected requests: {dq['rejected_requests']} ({dq['rejection_rate']:.1%})"
    )


def main() -> None:
    st.set_page_config(page_title="Aurynix Apex", page_icon=LOGO, layout="wide")
    st.logo(LOGO, size="large")
    st.title("Aurynix Apex: lead scoring")
    try:
        with st.spinner("API is starting up, please wait. This can take up to 2 minutes..."):
            info = wait_for_api("/model/info", API_WAKE_SECONDS)
    except ApiUnavailable as err:
        hint = (
            "It may still be starting: refresh this page in a minute."
            if API_WAKE_SECONDS
            else "Start it with `make run` (or `make docker-up`)."
        )
        st.error(f"{err} {hint}")
        return
    except RuntimeError as err:
        st.error(str(err))
        return
    test = info["test_metrics"]
    st.sidebar.markdown(
        f"**Model** `{info['model_version']}`  \n"
        f"Trained {info['trained_at'][:10]} on {info['training_rows']:,} leads  \n"
        f"Test PR-AUC **{test['pr_auc']:.3f}**  \n"
        f"Top 20%: precision **{test['precision_top20']:.0%}**, "
        f"recall **{test['recall_top20']:.0%}**"
    )
    st.sidebar.caption(f"API: {API_URL}")

    single, batch, monitoring = st.tabs(["Score a lead", "Score a CSV", "Monitoring"])
    tabs = [
        (single, tab_single),
        (batch, lambda: tab_batch(info["max_batch_size"])),
        (monitoring, tab_monitoring),
    ]
    for tab, render in tabs:
        with tab:
            try:
                render()
            except (RuntimeError, ValueError) as err:
                st.error(str(err))


if __name__ == "__main__":
    main()
