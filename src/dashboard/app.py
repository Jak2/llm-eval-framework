from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000/api")
API_KEY = os.getenv("API_KEY", "dev-secret-key")
HEADERS = {"X-API-Key": API_KEY}

st.set_page_config(page_title="LLM Eval Framework", page_icon="🧪", layout="wide")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 LLM Eval")
    days = st.slider("Time window (days)", 1, 30, 7)
    model_filter = st.selectbox("Filter by model", ["All", "claude", "openai"])
    st.divider()
    if st.button("🔄 Refresh data"):
        st.cache_data.clear()
        st.rerun()


# ── Data loaders (cached 30s) ─────────────────────────────────────────────────
@st.cache_data(ttl=30)
def load_summary(days: int) -> dict:
    r = requests.get(f"{API_BASE}/dashboard/summary", headers=HEADERS, params={"days": days}, timeout=10)
    return r.json() if r.ok else {}


@st.cache_data(ttl=30)
def load_trends(days: int, model: str) -> pd.DataFrame:
    params: dict = {"days": days}
    if model != "All":
        params["llm"] = model
    r = requests.get(f"{API_BASE}/dashboard/trends", headers=HEADERS, params=params, timeout=10)
    return pd.DataFrame(r.json()) if r.ok else pd.DataFrame()


@st.cache_data(ttl=30)
def load_regressions() -> list:
    r = requests.get(f"{API_BASE}/dashboard/regressions", headers=HEADERS, timeout=10)
    return r.json() if r.ok else []


@st.cache_data(ttl=30)
def load_results(limit: int = 100) -> pd.DataFrame:
    r = requests.get(f"{API_BASE}/results", headers=HEADERS, params={"limit": limit}, timeout=10)
    return pd.DataFrame(r.json()) if r.ok else pd.DataFrame()


summary = load_summary(days)
trends_df = load_trends(days, model_filter)
regressions = load_regressions()


# ── Headline Metrics ──────────────────────────────────────────────────────────
st.title("LLM Evaluation Framework")
st.caption("Real-time quality monitoring for your LLM outputs")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Runs", summary.get("total_runs", 0))
c2.metric("Pass Rate", f"{summary.get('pass_rate', 0):.0%}")
c3.metric("Avg Latency", f"{summary.get('avg_latency_ms', 0):.0f} ms")
c4.metric("Avg Judge Score", f"{summary.get('avg_judge_score', 0):.2f}")
c5.metric(
    "Regressions",
    len(regressions),
    delta=f"-{len(regressions)}" if regressions else None,
    delta_color="inverse",
)
st.divider()


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends", "📋 Results", "⚠️ Regressions", "🚀 Submit"])

with tab1:
    if not trends_df.empty:
        fig = px.line(
            trends_df,
            x="date", y="score", color="evaluator",
            title="Evaluator Scores Over Time",
            labels={"score": "Score (0–1)", "date": "Date"},
            markers=True,
        )
        fig.add_hline(y=0.7, line_dash="dash", line_color="red", annotation_text="Threshold (0.7)")
        fig.update_layout(legend_title="Evaluator", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No trend data yet. Submit test cases to populate this chart.")

with tab2:
    results_df = load_results()
    if not results_df.empty:
        display_cols = [c for c in
            ["id", "test_case_id", "overall_passed", "latency_ms", "token_count", "evaluated_at"]
            if c in results_df.columns]
        st.dataframe(results_df[display_cols], use_container_width=True, hide_index=True)
    else:
        st.info("No results yet.")

with tab3:
    if regressions:
        for reg in regressions:
            with st.expander(
                f"⚠️ {reg['test_case_name']} — {reg['evaluator']} dropped {reg['drop']:.2f}",
                expanded=True,
            ):
                col1, col2 = st.columns(2)
                col1.metric("Previous score", f"{reg['prev_score']:.2f}")
                col2.metric(
                    "Current score", f"{reg['curr_score']:.2f}",
                    delta=f"{-reg['drop']:.2f}", delta_color="inverse",
                )
                st.write(f"**Explanation:** {reg['explanation']}")
    else:
        st.success("✅ No regressions detected in the selected window.")

with tab4:
    st.subheader("Submit a New Test Case")
    with st.form("submit_form", clear_on_submit=True):
        name = st.text_input("Test case name *")
        prompt = st.text_area("Prompt *")
        col1, col2 = st.columns(2)
        llm = col1.selectbox("LLM", ["claude", "openai"])
        temp = col2.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
        evals = st.multiselect(
            "Evaluators",
            ["llm_judge", "hallucination", "faithfulness", "consistency"],
            default=["llm_judge"],
        )
        system_prompt = st.text_area("System prompt (optional)")
        reference = st.text_area("Reference answer (optional — enables hallucination detection)")
        context = st.text_area("RAG context (optional — enables faithfulness scoring)")
        submitted = st.form_submit_button("▶ Submit & Evaluate")

    if submitted:
        if not name or not prompt:
            st.error("Name and prompt are required.")
        else:
            payload: dict = {
                "name": name, "prompt": prompt, "llm_name": llm,
                "evaluators": evals, "temperature": temp,
            }
            if system_prompt:
                payload["system_prompt"] = system_prompt
            if reference:
                payload["reference_answer"] = reference
            if context:
                payload["context"] = context

            r = requests.post(f"{API_BASE}/test-cases", json=payload, headers=HEADERS, timeout=15)
            if r.ok:
                st.success(f"✅ Test case submitted! ID: {r.json()['id']} — evaluating in background.")
                st.cache_data.clear()
            else:
                st.error(f"Error {r.status_code}: {r.text}")
