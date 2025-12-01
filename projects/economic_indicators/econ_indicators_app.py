import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Economic Indicators Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("📊 Economic Indicators Dashboard")
st.markdown("Explore and visualize key economic indicators")

# Sidebar for controls
st.sidebar.header("Settings")

# Function to load data
@st.cache_data
def load_data(file_path):
    """Load economic data from CSV or other sources"""
    try:
        df = pd.read_csv('/Users/jessicabean/Library/CloudStorage/OneDrive-Personal/porftfolio.github.io/projects/economic_indicators/data/raw/economic_indicators_clean.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# File uploader
uploaded_file = st.sidebar.file_uploader(
    "Upload your economic data (CSV)", 
    type=['csv']
)

if uploaded_file is not None:
    # Load the data
    df = load_data(uploaded_file)
    
    if df is not None:
        # Display basic info
        st.subheader("Data Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            st.metric("Date Range", f"{df.iloc[0, 0]} - {df.iloc[-1, 0]}" if len(df) > 0 else "N/A")
        
        # Show raw data
        with st.expander("View Raw Data"):
            st.dataframe(df, use_container_width=True)
        
        # Select indicators to visualize
        st.subheader("Visualization")
        
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        if len(numeric_columns) > 0:
            selected_indicators = st.multiselect(
                "Select indicators to visualize:",
                numeric_columns,
                default=numeric_columns[:min(3, len(numeric_columns))]
            )
            
            if selected_indicators and len(df.columns) > 0:
                # Assume first column is date
                date_column = df.columns[0]
                
                # Create line chart
                fig = go.Figure()
                
                for indicator in selected_indicators:
                    fig.add_trace(go.Scatter(
                        x=df[date_column],
                        y=df[indicator],
                        mode='lines',
                        name=indicator
                    ))
                
                fig.update_layout(
                    title="Economic Indicators Over Time",
                    xaxis_title="Date",
                    yaxis_title="Value",
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistical summary
                st.subheader("Statistical Summary")
                st.dataframe(df[selected_indicators].describe(), use_container_width=True)
                
                # Correlation matrix
                if len(selected_indicators) > 1:
                    st.subheader("Correlation Matrix")
                    corr_matrix = df[selected_indicators].corr()
                    
                    fig_corr = px.imshow(
                        corr_matrix,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale='RdBu_r',
                        title="Correlation Between Indicators"
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.warning("No numeric columns found in the dataset")

else:
    st.info("👈 Please upload a CSV file with economic indicators to get started")
    
    # Example data structure
    st.subheader("Expected Data Format")
    st.markdown("""
    Your CSV should have a structure like:
    
    | Date       | GDP  | Unemployment | Inflation | ... |
    |------------|------|--------------|-----------|-----|
    | 2020-01-01 | 100  | 5.2          | 2.1       | ... |
    | 2020-02-01 | 102  | 5.0          | 2.3       | ... |
    
    - First column: Date/Time period
    - Other columns: Numeric economic indicators
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Upload your economic data CSV to start exploring!")
