import io
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap, LogNorm
import numpy as np
import pandas as pd
import scipy.stats as stats
import streamlit as st
from google import genai
from google.genai import types

from core.processor import ingest_and_process_streamflow
from core.plotters import (
    plot_mod01_efdc,
    plot_mod02_lag1,
    plot_mod03_matrix,
    plot_mod04_forecast,
    plot_mod05_antecedent,
    plot_mod06_persistence,
    plot_mod07_raster,
    plot_mod08_volumetric_raster,
    plot_mod09_spaghetti,
    plot_mod10_chrono_thresholds,
    plot_mod11_vol_thresholds,
    plot_mod12_composite
)

st.set_page_config(page_title="AI Hydroinformatics Studio", layout="wide")

# Safe API key retrieval (handles missing secrets gracefully)
api_key = None
try:
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

if not api_key:
    api_key = st.sidebar.text_input("Enter Gemini API Key (Required for AI Customization)", type="password")

st.title("AI Hydroinformatics Operational Studio")
st.caption("Visual Data Analytics, LLC — Deterministic Hydrological Visualization Pipeline")

# Initialize session state for iterative code prototyping
if "current_code" not in st.session_state:
    st.session_state.current_code = None
if "current_fig" not in st.session_state:
    st.session_state.current_fig = None
if "processed_df" not in st.session_state:
    st.session_state.processed_df = None

def extract_code(text: str) -> str:
    """Extracts executable python code block from model response."""
    match = re.search(r"```python\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1)
    match = re.search(r"```\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1)
    return text

def execute_code(code: str, df: pd.DataFrame):
    """Executes Matplotlib code in a safe local namespace."""
    local_env = {
        "df": df.copy(),
        "plt": plt,
        "ticker": ticker,
        "np": np,
        "pd": pd,
        "stats": stats,
        "ListedColormap": ListedColormap,
        "LogNorm": LogNorm
    }
    exec(code, local_env)
    return local_env.get("fig", plt.gcf())

col_left, col_right = st.columns([1, 2], gap="large")

with col_left:
    st.subheader("1. Ingest Dataset")
    uploaded_file = st.file_uploader("Upload Daily Streamflow CSV", type=["csv"])
    
    if uploaded_file:
        raw_df = pd.read_csv(uploaded_file)
        st.write("Column Mapping:")
        cols = list(raw_df.columns)
        date_col = st.selectbox("Date Column", cols, index=0)
        q_col = st.selectbox("Discharge Column (Q)", cols, index=1 if len(cols) > 1 else 0)
        station_label = st.text_input("Station / Gage Name", value="USGS Station")
        
        st.subheader("2. Operational Module")
        selected_mod = st.selectbox(
            "Select Module",
            [
                "MOD-01: Enhanced Flow Duration Curve (eFDC)",
                "MOD-02: Lag-1 Differential Hydrograph Point Cloud",
                "MOD-03: Discrete Transition Matrix (Raw Counts)",
                "MOD-04: Conditional Forecast Probability Matrix (Column-Normalized)",
                "MOD-05: Conditional Antecedent Probability Matrix (Row-Normalized)",
                "MOD-06: Sequential Flow Duration & Persistence Analysis",
                "MOD-07: Chronological Raster Hydrograph",
                "MOD-08: Volumetric Raster Hydrograph",
                "MOD-09: Annual FDC Spaghetti Plot",
                "MOD-10: Annual FDC Threshold Trends (Chronological)",
                "MOD-11: Annual FDC Volumetric Thresholds (Rank-Ordered)",
                "MOD-12: Composite Hydroinformatics Dashboard",
            ]
        )
        
        # Colormap options (relevant for raster & spaghetti plots)
        all_cmaps = sorted(list(plt.colormaps()))
        cmap_mode = st.radio("Colormap Mode", ["Curated Presets", "Custom Matplotlib Name"], horizontal=True)
        if cmap_mode == "Curated Presets":
            selected_cmap = st.selectbox("Preset Colormaps", ["USGS Discrete", "Spectral_r", "viridis", "cividis", "Blues"])
        else:
            custom_name = st.text_input("Matplotlib Colormap Name", value="Spectral_r")
            if custom_name in all_cmaps:
                selected_cmap = custom_name
            else:
                st.warning(f"'{custom_name}' not found. Using 'Spectral_r'.")
                selected_cmap = "Spectral_r"
                
        # Enforce prohibition of non-uniform colormaps per operational standards
        if selected_cmap.lower() in ["jet", "rainbow"]:
            st.error("Per operational rules, 'jet' and 'rainbow' are prohibited due to non-uniform perceptual gradients.")
            selected_cmap = "Spectral_r"

        run_btn = st.button("Generate Baseline Figure", type="primary", use_container_width=True)

with col_right:
    # 1. Baseline Generation
    if uploaded_file and run_btn:
        with st.spinner("Processing hydrology and verifying physical bounds..."):
            processed_df = ingest_and_process_streamflow(raw_df, date_col, q_col)
            st.session_state.processed_df = processed_df
            
            # Module Dispatcher
            if "MOD-01" in selected_mod:
                fig, code = plot_mod01_efdc(processed_df, station_name=station_label)
            elif "MOD-02" in selected_mod:
                fig, code = plot_mod02_lag1(processed_df, station_name=station_label)
            elif "MOD-03" in selected_mod:
                fig, code = plot_mod03_matrix(processed_df, station_name=station_label)
            elif "MOD-04" in selected_mod:
                fig, code = plot_mod04_forecast(processed_df, station_name=station_label)
            elif "MOD-05" in selected_mod:
                fig, code = plot_mod05_antecedent(processed_df, station_name=station_label)
            elif "MOD-06" in selected_mod:
                fig, code = plot_mod06_persistence(processed_df, station_name=station_label)
            elif "MOD-07" in selected_mod:
                fig, code = plot_mod07_raster(processed_df, station_name=station_label, cmap=selected_cmap)
            elif "MOD-08" in selected_mod:
                fig, code = plot_mod08_volumetric_raster(processed_df, station_name=station_label, cmap=selected_cmap)
            elif "MOD-09" in selected_mod:
                fig, code = plot_mod09_spaghetti(processed_df, station_name=station_label, cmap=selected_cmap)
            elif "MOD-10" in selected_mod:
                fig, code = plot_mod10_chrono_thresholds(processed_df, station_name=station_label)
            elif "MOD-11" in selected_mod:
                fig, code = plot_mod11_vol_thresholds(processed_df, station_name=station_label)
            elif "MOD-12" in selected_mod:
                fig, code = plot_mod12_composite(processed_df, station_name=station_label)
                
            st.session_state.current_fig = fig
            st.session_state.current_code = code

    # 2. Display and AI Modification Loop
    if st.session_state.current_fig is not None:
        st.pyplot(st.session_state.current_fig)
        
        # 300 DPI publication download button
        img_buf = io.BytesIO()
        st.session_state.current_fig.savefig(img_buf, format="png", dpi=300, bbox_inches="tight")
        st.download_button(
            label="Download 300 DPI Publication PNG",
            data=img_buf.getvalue(),
            file_name=f"{selected_mod[:6]}_{station_label.replace(' ', '_')}.png",
            mime="image/png"
        )
        
        st.markdown("---")
        st.subheader("💡 AI Plot Customization & Rapid Prototyping")
        st.caption("Enter natural language instructions to dynamically modify axes, scales, colors, or thresholds.")
        
        c_prompt, c_btn = st.columns([4, 1])
        with c_prompt:
            user_prompt = st.text_input(
                "Customization instruction:",
                placeholder="e.g., 'Change x-axis to a normal probability scale' or 'Add a horizontal line at median flow'"
            )
        with c_btn:
            st.write("")
            modify_btn = st.button("Apply AI Edit", type="primary", use_container_width=True)
            
        if modify_btn and user_prompt:
            if not api_key:
                st.error("Please enter a Gemini API Key in the left sidebar to enable AI customization.")
            else:
                with st.spinner("Applying AI customization to plotting code..."):
                    client = genai.Client(api_key=api_key)
                    system_prompt = (
                        "You are an expert hydroinformatics Python engineer adhering to strict Visual Data Analytics standards.\n"
                        "You are given existing Matplotlib code operating on a DataFrame named `df`.\n"
                        "Modify the code to fulfill the user's customization request.\n"
                        "Strict requirements:\n"
                        "1. Clean sans-serif typography (DejaVu Sans).\n"
                        "2. Zero-incursion legend rule: All legends and annotations must remain strictly outside the plotting frame.\n"
                        "3. Standard base-10 numerical labels on log scales (no scientific notation).\n"
                        "4. Output ONLY executable Python code inside a ```python ``` block.\n"
                        "5. Ensure the final Figure object is assigned to variable `fig`."
                    )
                    
                    full_prompt = (
                        "Existing Verified Code:\n"
                        "```python\n"
                        + str(st.session_state.current_code)
                        + "\n```\n\n"
                        + "User Customization Request:\n"
                        + f'"{user_prompt}"\n\n'
                        + "Return the complete, rewritten Python code applying this modification."
                    )
                    
                   # Robust generation handler with retry logic
import time

# Robust generation handler with retry logic
max_retries = 3
response = None

with st.spinner("AI is adapting the plot code..."):
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    # keep your existing config parameters here
                )
            )
            break
        except Exception as e:
            err_str = str(e)
            # If server is overloaded (503/500) or rate-limited (429), retry after a short pause
            if any(code in err_str for code in ["503", "500", "429", "UNAVAILABLE"]) and attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))  # waits 2s, then 4s
                continue
            else:
                st.error("The AI service is experiencing high traffic. Please wait a few seconds and click 'Apply AI Edit' again.")
                st.stop()

                    
                    new_code = extract_code(response.text)
                    try:
                        updated_fig = execute_code(new_code, st.session_state.processed_df)
                        st.session_state.current_fig = updated_fig
                        st.session_state.current_code = new_code
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error executing AI modified code: {e}")
                        with st.expander("Inspect Generated Code"):
                            st.code(new_code, language="python")
