import os
import time
import re
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Google GenAI SDK
from google import genai
from google.genai import types

# Page Configuration
st.set_page_config(
    page_title="AI Hydroinformatics Studio",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🌊 AI Hydroinformatics Studio")
st.markdown("Interactive streamflow data analytics and AI-assisted visualization prototyping.")

# Initialize Gemini Client via Streamlit Secrets or Environment Variable
api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))
if not api_key:
    st.error("Missing Gemini API Key. Please configure `GEMINI_API_KEY` in Streamlit Secrets.")
    st.stop()

client = genai.Client(api_key=api_key)

# ---------------------------------------------------------
# Sidebar: Data Ingestion & Module Selection
# ---------------------------------------------------------
with st.sidebar:
    st.header("1. Data Ingestion")
    uploaded_file = st.file_uploader("Upload Streamflow Time Series (CSV)", type=["csv"])
    
    st.header("2. Analytical Module")
    module_choice = st.selectbox(
        "Select Plot Routine",
        [
            "MOD-01: Enhanced Flow Duration Curve (eFDC)",
            "MOD-02: Lag-1 Differential Scatterplot (dQ/dt)",
            "MOD-03: Chronological Streamflow Hydrograph",
        ]
    )

    st.header("3. Colormap Settings")
    colormap_mode = st.radio(
        "Colormap Mode",
        ["Curated Presets", "Custom Matplotlib Name"]
    )
    if colormap_mode == "Curated Presets":
        selected_cmap = st.selectbox(
            "Preset Colormaps",
            ["USGS Discrete", "Spectral_r", "viridis", "plasma", "coolwarm"]
        )
    else:
        selected_cmap = st.text_input("Matplotlib Colormap Name", value="viridis")

# ---------------------------------------------------------
# Data Loading & Preprocessing
# ---------------------------------------------------------
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Identify Date & Flow Columns
        date_candidates = [col for col in df.columns if any(k in col.lower() for k in ["date", "time", "datetime"])]
        flow_candidates = [col for col in df.columns if any(k in col.lower() for k in ["flow", "discharge", "q", "streamflow", "cfs", "cms"])]
        
        date_col = date_candidates[0] if date_candidates else df.columns[0]
        flow_col = flow_candidates[0] if flow_candidates else (df.columns[1] if len(df.columns) > 1 else df.columns[0])
        
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df[flow_col] = pd.to_numeric(df[flow_col], errors="coerce")
        df = df.dropna(subset=[date_col, flow_col]).sort_values(by=date_col).reset_index(drop=True)
        
        st.session_state["df"] = df
        st.session_state["date_col"] = date_col
        st.session_state["flow_col"] = flow_col
    except Exception as e:
        st.error(f"Error parsing uploaded CSV: {e}")
        st.stop()
else:
    st.info("Please upload a daily streamflow CSV file in the sidebar to begin.")
    st.stop()

# ---------------------------------------------------------
# Baseline Script Templates
# ---------------------------------------------------------
def generate_baseline_code(module_name, date_c, flow_c, cmap):
    if "MOD-01" in module_name:
        return f"""import matplotlib.pyplot as plt
import numpy as np

# Prepare Exceedance Probability for Enhanced Flow Duration Curve (eFDC)
sorted_q = np.sort(df['{flow_c}'])[::-1]
n = len(sorted_q)
exceedance_prob = (np.arange(1, n + 1) / (n + 1)) * 100.0

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
ax.plot(exceedance_prob, sorted_q, color='navy', lw=2, label='Observed Streamflow')

ax.set_yscale('log')
ax.set_xlabel('Exceedance Probability (%)', fontsize=11)
ax.set_ylabel('Discharge ({flow_c})', fontsize=11)
ax.set_title('MOD-01: Enhanced Flow Duration Curve (eFDC)', fontsize=13, weight='bold')

ax.grid(True, which='major', linestyle='-', linewidth=0.75, alpha=0.7)
ax.grid(True, which='minor', linestyle='--', linewidth=0.5, alpha=0.5)
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)
"""
    elif "MOD-02" in module_name:
        return f"""import matplotlib.pyplot as plt
import numpy as np

q_t = df['{flow_c}'][:-1].values
q_t1 = df['{flow_c}'][1:].values

fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
sc = ax.scatter(q_t, q_t1, c=np.arange(len(q_t)), cmap='{cmap}', alpha=0.6, s=15, edgecolors='none')

# 1:1 Identity / Equilibrium line
min_val = max(1e-2, min(q_t.min(), q_t1.min()))
max_val = max(q_t.max(), q_t1.max())
ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5, label='1:1 Equilibrium Line')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Discharge Q(t)', fontsize=11)
ax.set_ylabel('Discharge Q(t+1)', fontsize=11)
ax.set_title('MOD-02: Lag-1 Differential Phase Scatterplot', fontsize=13, weight='bold')

ax.grid(True, which='major', linestyle='-', linewidth=0.75, alpha=0.7)
ax.grid(True, which='minor', linestyle='--', linewidth=0.5, alpha=0.5)
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)
"""
    else:  # MOD-03
        return f"""import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(11, 5), dpi=300)
ax.plot(df['{date_c}'], df['{flow_c}'], color='royalblue', lw=1.2, label='Daily Discharge')

ax.set_xlabel('Date', fontsize=11)
ax.set_ylabel('Discharge ({flow_c})', fontsize=11)
ax.set_title('MOD-03: Chronological Streamflow Hydrograph', fontsize=13, weight='bold')

ax.grid(True, which='major', linestyle='-', linewidth=0.75, alpha=0.7)
ax.grid(True, which='minor', linestyle='--', linewidth=0.5, alpha=0.5)
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)
"""

# Initialize or reset the active script
if "current_code" not in st.session_state or st.sidebar.button("Generate Baseline Figure"):
    st.session_state["current_code"] = generate_baseline_code(
        module_choice, st.session_state["date_col"], st.session_state["flow_col"], selected_cmap
    )

# ---------------------------------------------------------
# Execution & Display
# ---------------------------------------------------------
st.subheader("Visual Output")
try:
    exec_scope = {
        "df": st.session_state["df"],
        "pd": pd,
        "plt": plt,
        "st": st,
    }
    exec(st.session_state["current_code"], exec_scope)
except Exception as e:
    st.error(f"Execution Error in Generated Plotting Routine: {e}")

# ---------------------------------------------------------
# AI Customization Interface
# ---------------------------------------------------------
st.markdown("---")
st.subheader("💡 AI Plot Customization & Rapid Prototyping")
st.caption("Enter natural language instructions to dynamically modify axes, scales, colors, or thresholds.")

col_input, col_btn = st.columns([4, 1])
with col_input:
    custom_instruction = st.text_input(
        "Customization instruction:",
        placeholder="e.g., Modify the x-axis to major intervals of 10%, or filter for summer months only."
    )
with col_btn:
    st.write("")
    st.write("")
    apply_ai = st.button("Apply AI Edit", type="primary")

if apply_ai and custom_instruction:
    existing_code = st.session_state.get("current_code", "")
    date_col_name = st.session_state.get("date_col", "")
    flow_col_name = st.session_state.get("flow_col", "")

    full_prompt = (
        "You are an expert Python data visualization developer specializing in hydrology and Matplotlib.\n"
        "Below is an existing Streamlit Matplotlib script and a user instruction to modify the plot.\n\n"
        f"The dataframe `df` is already loaded with columns:\n"
        f"- Date column: '{date_col_name}' (datetime format)\n"
        f"- Flow column: '{flow_col_name}' (numeric discharge)\n\n"
        "Existing script:\n"
        "```python\n"
        + existing_code + "\n"
        "```\n\n"
        f"User instruction:\n\"{custom_instruction}\"\n\n"
        "Requirements:\n"
        "1. Modify the script to satisfy the instruction cleanly.\n"
        "2. Maintain publication quality: Arial/DejaVu Sans style, external legend placement, standard gridlines.\n"
        "3. Must render via `st.pyplot(fig)` and end with `plt.close(fig)`.\n"
        "4. Output ONLY valid, executable Python code enclosed within standard ```python ``` blocks. No introductory or trailing markdown prose."
    )

    max_retries = 3
    response = None

    with st.spinner("AI is adapting the plot code..."):
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                    )
                )
                break
            except Exception as e:
                err_msg = str(e)
                if any(code in err_msg for code in ["503", "500", "429", "UNAVAILABLE"]) and attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                    continue
                else:
                    st.error("The AI service is experiencing high traffic. Please wait a few seconds and click 'Apply AI Edit' again.")
                    st.stop()

    if response and response.text:
        extracted = re.search(r"```python\s*(.*?)\s*```", response.text, re.DOTALL)
        clean_code = extracted.group(1) if extracted else response.text.replace("```", "").strip()
        st.session_state["current_code"] = clean_code
        st.rerun()

# ---------------------------------------------------------
# Code Drawer / Debug Inspector
# ---------------------------------------------------------
with st.expander("Inspect Current Python Routine"):
    st.code(st.session_state.get("current_code", ""), language="python")
