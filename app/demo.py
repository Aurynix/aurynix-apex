"""Streamlit demo for Aurynix Apex.

    make run    # the API must be running first
    make demo   # → http://localhost:8501

The demo is a client of the API (`API_URL`, default http://localhost:8000):
it never loads the model itself. Three tabs: score one lead, score a CSV,
and drift monitoring.
"""

import json
import os
import urllib.error
import urllib.request

import pandas as pd
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")

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


def call(method: str, path: str, payload: dict | None = None) -> dict:
    """Call the API and return its JSON; raises RuntimeError with a readable message."""
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        f"{API_URL}{path}", data=data, method=method, headers={"content-type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as err:
        raise RuntimeError(f"API error {err.code}: {err.read().decode()[:500]}") from err
    except urllib.error.URLError as err:
        raise RuntimeError(f"Cannot reach the API at {API_URL} ({err.reason}).") from err


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


def tab_batch() -> None:
    st.caption(
        "Upload a CSV with the API fields (`lead_origin`, `lead_source`, `do_not_email`, "
        "`total_visits`, `time_on_website`, `specialization`, `occupation`) or the raw "
        "Kaggle columns (e.g. `Leads.csv`). Up to 10,000 rows."
    )
    file = st.file_uploader("Leads CSV", type="csv")
    if not file:
        return
    df = pd.read_csv(file).head(10_000)
    leads = leads_from_csv(df)
    result = call("POST", "/predict/batch", {"leads": leads})["predictions"]
    ranked = pd.DataFrame(
        {
            "score": [p["score"] for p in result],
            "segment": [SEGMENT_ICON[p["segment"]] for p in result],
            "top reason up": [", ".join(p["reasons"]["up"][:1]) for p in result],
            "top reason down": [", ".join(p["reasons"]["down"][:1]) for p in result],
        },
        index=df.index,
    )
    if "Prospect ID" in df.columns:
        ranked.insert(0, "Prospect ID", df["Prospect ID"])
    ranked = ranked.sort_values("score", ascending=False)
    counts = ranked["segment"].value_counts()
    cols = st.columns(3)
    for col, seg in zip(cols, SEGMENT_ICON.values(), strict=True):
        col.metric(seg, f"{counts.get(seg, 0):,}")
    st.dataframe(ranked, width="stretch", hide_index=True)
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
    st.caption(f"{report['n_samples']:,} predictions · run at {report['created_at']}")
    if "features" not in report:
        st.warning(f"Not enough predictions for a reliable check (status: {report['status']}).")
        return

    shares = report["prediction"]["segment_shares"]
    st.subheader("Segments: training vs. now")
    st.bar_chart(pd.DataFrame(shares).rename(index=str.capitalize), stack=False)

    st.subheader("Feature drift (PSI)")
    features = pd.DataFrame(report["features"]).T.sort_values("psi", ascending=False)
    features["status"] = features["status"].map(STATUS_ICON)
    st.dataframe(features[["status", "psi"]], width="stretch")

    dq = report["data_quality"]
    st.subheader(f"Data quality: {STATUS_ICON.get(dq['status'], dq['status'])}")
    missing = pd.DataFrame(dq["missing_rate"]).T[["reference", "current", "status"]]
    st.dataframe(missing, width="stretch")
    st.caption(
        f"Leads relying on defaults: {dq['defaulted_share']:.0%} · "
        f"rejected requests: {dq['rejected_requests']} ({dq['rejection_rate']:.1%})"
    )


def main() -> None:
    st.set_page_config(page_title="Aurynix Apex", page_icon="🎯", layout="wide")
    st.title("🎯 Aurynix Apex: lead scoring")
    try:
        info = call("GET", "/model/info")
    except RuntimeError as err:
        st.error(f"{err} Start it with `make run` (or `make docker-up`).")
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
    for tab, render in [(single, tab_single), (batch, tab_batch), (monitoring, tab_monitoring)]:
        with tab:
            try:
                render()
            except (RuntimeError, ValueError) as err:
                st.error(str(err))


if __name__ == "__main__":
    main()
