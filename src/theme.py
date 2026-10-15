"""Visual system for the decision-support desk."""

from __future__ import annotations

import streamlit as st

CSS = """
<style>
  .stApp { background: #f4f0e8; }
  .block-container { padding-top: 1.1rem; padding-bottom: 3.5rem; max-width: 1220px; }
  h1, h2, h3 { letter-spacing: -0.025em; color: #1c1917; }
  [data-testid="stSidebar"] { background: #f3eee6; border-right: 1px solid #e6dfd4; }
  [data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }
  div[data-testid="stMetric"] {
    background: #fffdf9;
    border: 1px solid #e6dfd4;
    border-top: 3px solid #9a3412;
    padding: 0.55rem 0.8rem 0.15rem;
    border-radius: 12px;
    box-shadow: 0 1px 0 rgba(28, 25, 23, 0.03);
  }
  div[data-testid="stMetric"] label { color: #57534e !important; font-size: 0.82rem !important; }
  div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #1c1917; font-variant-numeric: tabular-nums; }
  .app-header {
    display: flex; gap: 14px; align-items: flex-start;
    padding: 2px 0 14px; margin-bottom: 8px;
    border-bottom: 1px solid #e6dfd4;
  }
  .brand-mark {
    width: 46px; height: 46px; border-radius: 12px;
    background: #1c1917; color: #f6f1e7;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; letter-spacing: 0.04em; flex: 0 0 auto;
  }
  .eyebrow { font-size: 0.75rem; letter-spacing: 0.12em; text-transform: uppercase; color: #9a3412; font-weight: 700; }
  .app-title { font-size: 1.45rem; font-weight: 720; color: #1c1917; line-height: 1.15; margin-top: 2px; }
  .app-sub { color: #57534e; font-size: 0.92rem; margin-top: 4px; max-width: 760px; }
  .page-intro { margin: 8px 0 14px; }
  .page-intro h1 { font-size: 1.55rem; margin: 0 0 4px; }
  .kicker { font-size: 0.74rem; letter-spacing: 0.12em; text-transform: uppercase; color: #9a3412; font-weight: 700; }
  .lede { color: #44403c; font-size: 1rem; margin: 0; max-width: 860px; line-height: 1.45; }
  .callout { border-radius: 12px; padding: 12px 14px; margin: 8px 0 12px; border: 1px solid; }
  .callout-critical { background: #fff1f2; border-color: #fecdd3; }
  .callout-warning { background: #fff7ed; border-color: #fed7aa; }
  .callout-info { background: #f0fdfa; border-color: #99f6e4; }
  .callout-neutral { background: #fffdf9; border-color: #e6dfd4; }
  .callout-kicker { font-size: 0.72rem; letter-spacing: 0.1em; text-transform: uppercase; font-weight: 700; color: #57534e; }
  .callout-title { font-weight: 700; color: #1c1917; margin: 2px 0 4px; }
  .callout-body { color: #44403c; line-height: 1.45; font-size: 0.95rem; }
  .badge { display: inline-block; border-radius: 999px; padding: 2px 8px; font-size: 0.75rem; font-weight: 700; border: 1px solid; margin-right: 6px; }
  .badge-critical { background: #fff1f2; color: #9f1239; border-color: #fecdd3; }
  .badge-warning { background: #fff7ed; color: #9a3412; border-color: #fed7aa; }
  .badge-info { background: #f0fdfa; color: #115e59; border-color: #99f6e4; }
  .badge-neutral { background: #f5f5f4; color: #44403c; border-color: #e7e5e4; }
  .foot { color: #78716c; font-size: 0.8rem; margin-top: 22px; }
  .weight-row { display: flex; gap: 8px; flex-wrap: wrap; margin: 6px 0 10px; }
  .weight-chip { background: #fffdf9; border: 1px solid #e6dfd4; border-radius: 999px; padding: 4px 10px; font-size: 0.82rem; color: #44403c; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
