import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

#import warnings
#warnings.filterwarnings('ignore')

#print("✅ All libraries imported successfully!")

# ============================================
# STEP 1: IMPORT DATA
# ===========================================
economic_data = pd.read_csv("/Users/jessicabean/Library/CloudStorage/OneDrive-Personal/porftfolio.github.io/projects/economic_indicators/data/raw/economic_data.csv")
# ============================================
# STEP 3: RUN PREPROCESSING STEPS
# ============================================
economic_data = fetch_economic_data(start_date='2015-01-01')

# Save to CSV
df_clean.to_csv("df_clean.csv", index=False)
