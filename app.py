"""
This module is a Streamlit application that utilizes Google GenAI for benchmarking large language models.
It imports necessary libraries for data manipulation (pandas), visualization (plotly.express), and time management.

Modules:
- streamlit: For creating the web application interface.
- google.genai: For accessing Google’s Generative AI functionalities.
- time: For handling time-related tasks.
- pandas: For data manipulation and analysis.
- plotly.express: For creating interactive visualizations.

Filepath:
- /home/chisom/codex/llm-benchmarking/app.py
"""

import streamlit as st
from google import genai
import time
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="LLM Benchmarking", layout="wide")

st.title("LLM Benchmarking")
st.subheader("Compare as many LLMs as you want side by side")
st.divider()