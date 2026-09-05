"""
Streamlit UI for the Corporate Policy & Compliance Assistant.
Optional interface (Section 8.2 Good-to-Have) to exercise the pipeline
with citation highlighting.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src" / "generation"))
sys.path.insert(0, str(Path(__file__).parent / "src" / "retrieval"))
sys.path.insert(0, str(Path(__file__).parent / "src" / "ingestion"))

import streamlit as st
from generate_answer import answer_question

st.set_page_config(page_title="Solara Policy Assistant", page_icon="📋")

st.title("📋 Corporate Policy & Compliance Assistant")
st.caption("Ask a question about Solara Technologies Inc. policies. "
           "Answers are grounded strictly in the policy corpus.")

question = st.text_input("Your question:", placeholder="e.g. What is the gift limit?")

if st.button("Ask", type="primary") and question:
    with st.spinner("Retrieving relevant policy clauses and generating answer..."):
        try:
            result, _ = answer_question(question)
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

    if result.abstained:
        st.warning("⚠️ The policy corpus does not contain enough information "
                   "to answer this question confidently.")
        st.write(result.answer)
    else:
        st.success("Answer found")
        st.write(result.answer)

        st.subheader("📎 Citations")
        for c in result.citations:
            st.markdown(f"- `{c.doc_id}` — Clause `{c.clause_id}`")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Confidence", result.confidence.capitalize())
        with col2:
            st.metric("Approval Required", result.required_approval[:30] if result.required_approval != "None" else "None")

        with st.expander("Full details"):
            st.write(f"**Applicable Policy:** {result.applicable_policy}")
            st.write(f"**Rule/Limit:** {result.rule_or_limit}")
            st.write(f"**Required Approval:** {result.required_approval}")

st.divider()
st.caption("Built for the AAIE_025_LGL capstone — Solara Technologies Inc. "
           "(fictional company, synthetic data only)")