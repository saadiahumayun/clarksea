import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Page config
st.set_page_config(
    page_title="ClarkSea Index Analysis",
    page_icon="🚢",
    layout="wide"
)

# Custom CSS
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stPlotlyChart {
        background-color: white;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        padding: 1rem;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stMetric {
        background-color: white;
        padding: 1rem;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .analysis-box {
        background-color: #f8f9fa;
        border-left: 4px solid #2c3e50;
        padding: 1.5rem;
        border-radius: 0 5px 5px 0;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .market-analysis {
        background-color: #f1f8ff;
        border: 1px solid #cfe4ff;
        border-radius: 8px;
        padding: 20px;
        margin: 15px 0;
        width: 100%;
    }
    .market-extremes {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        border-radius: 8px;
        padding: 20px;
        margin: 15px 0;
        width: 100%;
        display: block;
    }
    .market-trends {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 20px;
        margin: 15px 0;
        width: 100%;
        display: block;
    }
    .seasonal-analysis {
        background-color: #e2e3e5;
        border: 1px solid #d6d8db;
        border-radius: 8px;
        padding: 20px;
        margin: 15px 0;
        width: 100%;
        display: block;
    }
    .volatility-insights {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 8px;
        padding: 20px;
        margin: 15px 0;
        width: 100%;
        display: block;
    }
    .cycle-analysis {
        background-color: #fff3e0;
        border: 1px solid #ffe0b2;
        border-radius: 8px;
        padding: 20px;
        margin: 15px 0;
        width: 100%;
        display: block;
    }
    /* Style for the new tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 24px;
        background-color: #ffffff;
        border-radius: 4px 4px 0 0;
        color: #2c3e50;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #f8f9fa;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #2c3e50;
        color: #ffffff;
    }
    .yoy-analysis {
        background-color: #e8eaf6;
        border: 1px solid #c5cae9;
        border-radius: 8px;
        padding: 20px;
        margin: 15px 0;
        width: 100%;
        display: block;
    }
    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    # Read excel file starting from row 5 (index 4)
    df = pd.read_excel(
        'SIN_Timeseries_20241227193839.xlsx', 
        skiprows=4,
        names=['date', 'value']  # Explicitly name the columns
    )
    
    # Convert the date column to datetime, handling any parsing errors
    try:
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
    except:
        try:
            # If first attempt fails, try parsing without specific format
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        except Exception as e:
            st.error(f"Error parsing dates: {e}")
            st.write("First few rows of date column:", df['date'].head())
            return None

    # Drop rows with invalid dates
    df = df.dropna(subset=['date'])
    
    # Ensure value column is numeric
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.dropna(subset=['value'])
    
    # Calculate additional metrics
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['month_name'] = df['date'].dt.strftime('%B')
    
    # Calculate rolling metrics with min_periods=1
    df['rolling_avg_52w'] = df['value'].rolling(window=52, min_periods=1).mean()
    df['yoy_change'] = df.groupby(df['date'].dt.month)['value'].pct_change(periods=52).fillna(0) * 100
    df['volatility'] = df['value'].rolling(window=12, min_periods=1).std()
    
    return df

# Load the data
df = load_data()

if df is None:
    st.error("Failed to load data properly. Please check the data format.")
    st.stop()

# Header
st.title("🚢 ClarkSea Index Dashboard")
st.markdown("Interactive analysis of shipping rates from 1990 to 2024")

# Sidebar filters
st.sidebar.header("Filters")
years_range = st.sidebar.slider(
    "Select Year Range",
    min_value=int(df['year'].min()),
    max_value=int(df['year'].max()),
    value=(int(df['year'].min()), int(df['year'].max()))
)

# Replace with tabs
tab1, tab2, tab3 = st.tabs(["Overview", "Trends & Patterns", "Market Cycles"])

# Filter data based on selection
df_filtered = df[(df['year'] >= years_range[0]) & (df['year'] <= years_range[1])]

# Replace if-elif-else structure with tab contexts
with tab1:
    # Overview Section
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Average Rate", f"${df_filtered['value'].mean():,.0f}/day")
    with col2:
        st.metric("Maximum Rate", f"${df_filtered['value'].max():,.0f}/day")
    with col3:
        st.metric("Minimum Rate", f"${df_filtered['value'].min():,.0f}/day")
    with col4:
        st.metric("Volatility", f"{df_filtered['value'].std():,.0f}")

    # Main time series chart
    st.subheader("Historical Rates")
    fig_main = go.Figure()
    
    fig_main.add_trace(go.Scatter(
        x=df_filtered['date'],
        y=df_filtered['value'],
        name='Daily Rate',
        line=dict(color='#1f77b4')
    ))
    
    fig_main.add_trace(go.Scatter(
        x=df_filtered['date'],
        y=df_filtered['rolling_avg_52w'],
        name='52-week MA',
        line=dict(color='#2ca02c', dash='dash')
    ))
    
    fig_main.update_layout(
        height=600,
        template='plotly_white',
        hovermode='x unified',
        yaxis_title='$/day',
        xaxis_title='Date',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    st.plotly_chart(fig_main, use_container_width=True)

    # Add analysis after main chart
    st.markdown('<div class="market-analysis"><h3>Market Analysis</h3>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        curr_mean = df_filtered['value'].mean()
        curr_max = df_filtered['value'].max()
        curr_max_date = df_filtered.loc[df_filtered['value'].idxmax(), 'date']
        curr_min = df_filtered['value'].min()
        curr_min_date = df_filtered.loc[df_filtered['value'].idxmin(), 'date']
        
        extremes_html = f"""
        <div class="market-extremes">
            <h4>Market Extremes</h4>
            <ul>
                <li><strong>Highest Rate:</strong> ${curr_max:,.0f}/day ({curr_max_date.strftime('%B %Y')})</li>
                <li><strong>Lowest Rate:</strong> ${curr_min:,.0f}/day ({curr_min_date.strftime('%B %Y')})</li>
                <li><strong>Average Rate:</strong> ${curr_mean:,.0f}/day</li>
            </ul>
        </div>
        """
        st.markdown(extremes_html, unsafe_allow_html=True)
    
    with col2:
        first_value = df_filtered.iloc[0]['value']
        last_value = df_filtered.iloc[-1]['value']
        total_change = ((last_value - first_value) / first_value) * 100
        avg_volatility = df_filtered['volatility'].mean()
        
        trend = 'Upward' if total_change > 0 else 'Downward'
        change_type = 'increase' if total_change > 0 else 'decrease'
        
        trends_html = f"""
        <div class="market-trends">
            <h4>Market Trends</h4>
            <ul>
                <li>Overall trend: {trend}</li>
                <li>Total change: {abs(total_change):.1f}% {change_type}</li>
                <li>Average volatility: ${avg_volatility:,.0f}</li>
            </ul>
        </div>
        """
        st.markdown(trends_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Year-over-year changes
    st.subheader("Year-over-Year Changes")
    # Create more detailed YoY change visualization
    fig_yoy = go.Figure()
    
    # Add YoY change line
    fig_yoy.add_trace(go.Scatter(
        x=df_filtered['date'],
        y=df_filtered['yoy_change'],
        mode='lines',
        name='YoY Change',
        line=dict(color='#1f77b4'),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.1)'
    ))
    
    # Add zero line reference
    fig_yoy.add_hline(
        y=0, 
        line_dash="dash", 
        line_color="red",
        annotation_text="No Change",
        annotation_position="bottom right"
    )
    
    fig_yoy.update_layout(
        height=400,
        template='plotly_white',
        yaxis_title='Year-over-Year Change (%)',
        xaxis_title='Date',
        hovermode='x unified',
        yaxis=dict(
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='rgba(0,0,0,0.2)',
            tickformat='.1f'
        )
    )
    st.plotly_chart(fig_yoy, use_container_width=True)

    # Add YoY analysis
    st.markdown("### Year-over-Year Analysis")
    yoy_pos = (df_filtered['yoy_change'] > 0).mean() * 100
    max_yoy = df_filtered['yoy_change'].max()
    min_yoy = df_filtered['yoy_change'].min()
    
    yoy_html = f"""
    <div class="yoy-analysis">
        <ul>
            <li>Positive YoY changes occurred in {yoy_pos:.1f}% of observations</li>
            <li>Largest YoY increase: {max_yoy:.1f}%</li>
            <li>Largest YoY decrease: {min_yoy:.1f}%</li>
        </ul>
    </div>
    """
    st.markdown(yoy_html, unsafe_allow_html=True)

with tab2:
    # Seasonal Pattern
    st.subheader("Seasonal Patterns")
    monthly_avg = df_filtered.groupby('month_name')['value'].mean().reset_index()
    monthly_avg['month_num'] = pd.to_datetime(monthly_avg['month_name'], format='%B').dt.month
    monthly_avg = monthly_avg.sort_values('month_num')
    
    fig_seasonal = go.Figure()
    fig_seasonal.add_trace(go.Scatterpolar(
        r=monthly_avg['value'],
        theta=monthly_avg['month_name'],
        fill='toself',
        name='Average Rate',
        line=dict(color='#1f77b4')
    ))
    
    fig_seasonal.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                showline=False,
                showticklabels=True,
                gridcolor="white",
                tickformat="$,.0f"
            ),
        ),
        showlegend=False,
        template='plotly_white',
        height=500
    )
    st.plotly_chart(fig_seasonal, use_container_width=True)

    # Add seasonal analysis
    st.markdown('<div class="seasonal-analysis"><h3>Seasonal Analysis</h3>', unsafe_allow_html=True)
    
    monthly_stats = df_filtered.groupby('month_name').agg({
        'value': ['mean', 'std']
    }).reset_index()
    monthly_stats.columns = ['month', 'mean', 'std']
    best_months = monthly_stats.nlargest(3, 'mean')
    worst_months = monthly_stats.nsmallest(3, 'mean')
    
    # Seasonal Analysis
    seasonal_html = f"""
    <div class="seasonal-analysis">
        <h3>Seasonal Analysis</h3>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
            <div>
                <h4>Strongest Months</h4>
                <ul>
                    {' '.join([f"<li>{month['month']}: ${month['mean']:,.0f}/day (±${month['std']:,.0f})</li>" for _, month in best_months.iterrows()])}
                </ul>
            </div>
            <div>
                <h4>Weakest Months</h4>
                <ul>
                    {' '.join([f"<li>{month['month']}: ${month['mean']:,.0f}/day (±${month['std']:,.0f})</li>" for _, month in worst_months.iterrows()])}
                </ul>
            </div>
        </div>
    </div>
    """
    st.markdown(seasonal_html, unsafe_allow_html=True)

    # Distribution by Year
    st.subheader("Annual Rate Distribution")
    # Create more detailed annual distribution
    yearly_stats = df_filtered.groupby('year').agg({
        'value': ['mean', 'std', 'min', 'max', 'median']
    }).reset_index()
    yearly_stats.columns = ['year', 'mean', 'std', 'min', 'max', 'median']
    
    fig_yearly = go.Figure()
    
    # Add box plot
    fig_yearly.add_trace(go.Box(
        x=df_filtered['year'],
        y=df_filtered['value'],
        name='Distribution',
        boxpoints='outliers',
        marker_color='#1f77b4',
        fillcolor='rgba(31, 119, 180, 0.5)'
    ))
    
    # Add mean line
    fig_yearly.add_trace(go.Scatter(
        x=yearly_stats['year'],
        y=yearly_stats['mean'],
        mode='lines',
        name='Annual Mean',
        line=dict(color='red', dash='dash')
    ))
    
    fig_yearly.update_layout(
        height=500,
        template='plotly_white',
        yaxis_title='Daily Rate ($/day)',
        xaxis_title='Year',
        showlegend=True,
        yaxis=dict(
            tickformat='$,.0f'
        )
    )
    st.plotly_chart(fig_yearly, use_container_width=True)

    # Monthly Trends Heatmap
    st.subheader("Monthly Patterns Over Years")
    pivot_data = df_filtered.pivot_table(
        values='value',
        index='year',
        columns='month_name',
        aggfunc='mean'
    )
    
    fig_heatmap = px.imshow(
        pivot_data,
        aspect='auto',
        color_continuous_scale='RdYlBu_r',
        labels=dict(x='Month', y='Year', color='$/day')
    )
    fig_heatmap.update_layout(height=400)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    # Volatility Analysis
    st.subheader("Market Volatility")
    
    # Add volatility controls
    vol_window = st.slider("Volatility Window (weeks)", 4, 52, 12)
    df_filtered['rolling_vol'] = df_filtered['value'].rolling(window=vol_window).std()
    
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(
        x=df_filtered['date'],
        y=df_filtered['rolling_vol'],
        fill='tozeroy',
        name=f'{vol_window}-week Volatility',
        line=dict(color='rgba(31, 119, 180, 0.8)')
    ))
    
    fig_vol.update_layout(
        height=400,
        template='plotly_white',
        yaxis_title='Standard Deviation',
        xaxis_title='Date',
        hovermode='x unified'
    )
    st.plotly_chart(fig_vol, use_container_width=True)

     # Add volatility analysis
    st.markdown('<div class="volatility-insights">', unsafe_allow_html=True)
    st.markdown("### Volatility Insights")
    recent_vol = df_filtered['rolling_vol'].tail(52).mean()
    historical_vol = df_filtered['rolling_vol'].mean()
    vol_change = ((recent_vol - historical_vol) / historical_vol) * 100
    
    volatility_html = f"""
    <div class="volatility-insights">
        <h3>Volatility Insights</h3>
        <ul>
            <li>Recent volatility (last year) is {abs(vol_change):.1f}% {'higher' if vol_change > 0 else 'lower'} than historical average</li>
            <li>Current volatility trend: {'Increasing' if vol_change > 0 else 'Decreasing'}</li>
            <li>Volatility tends to {'increase' if df_filtered['rolling_vol'].corr(df_filtered['value']) > 0 else 'decrease'} with higher rate levels</li>
        </ul>
    </div>
    """
    st.markdown(volatility_html, unsafe_allow_html=True)

    # Market Cycle Detection
    st.subheader("Market Cycles Analysis")
    
    cycle_window = st.slider("Cycle Detection Window (weeks)", 12, 52, 26)
    
    df_filtered['cycle_ma'] = df_filtered['value'].rolling(window=cycle_window).mean()
    df_filtered['cycle_std'] = df_filtered['value'].rolling(window=cycle_window).std()
    df_filtered['upper_band'] = df_filtered['cycle_ma'] + (2 * df_filtered['cycle_std'])
    df_filtered['lower_band'] = df_filtered['cycle_ma'] - (2 * df_filtered['cycle_std'])

    fig_cycles = go.Figure()
    
    # Add main rate line
    fig_cycles.add_trace(go.Scatter(
        x=df_filtered['date'],
        y=df_filtered['value'],
        name='Daily Rate',
        line=dict(color='#1f77b4')
    ))
    
    # Add bands
    fig_cycles.add_trace(go.Scatter(
        x=df_filtered['date'],
        y=df_filtered['upper_band'],
        fill=None,
        mode='lines',
        line_color='rgba(0,100,80,0.2)',
        name='Upper Band'
    ))
    
    fig_cycles.add_trace(go.Scatter(
        x=df_filtered['date'],
        y=df_filtered['lower_band'],
        fill='tonexty',
        mode='lines',
        line_color='rgba(0,100,80,0.2)',
        name='Lower Band'
    ))
    
    fig_cycles.update_layout(
        height=600,
        template='plotly_white',
        yaxis_title='$/day',
        xaxis_title='Date',
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_cycles, use_container_width=True)

    # Add cycle analysis
    st.markdown("### Market Cycle Characteristics")
    above_upper = (df_filtered['value'] > df_filtered['upper_band']).mean() * 100
    below_lower = (df_filtered['value'] < df_filtered['lower_band']).mean() * 100
    
    cycle_html = f"""
    <div class="cycle-analysis">
        <h3>Market Cycle Characteristics</h3>
        <ul>
            <li>Rates exceed upper band in {above_upper:.1f}% of observations</li>
            <li>Rates fall below lower band in {below_lower:.1f}% of observations</li>
            <li>The selected {cycle_window}-week window shows {'high' if above_upper + below_lower > 10 else 'normal'} market cycle activity</li>
        </ul>
    </div>
    """
    st.markdown(cycle_html, unsafe_allow_html=True)

    # Add return distribution
    st.subheader("Return Distribution")
    df_filtered['returns'] = df_filtered['value'].pct_change() * 100
    
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=df_filtered['returns'],
        nbinsx=50,
        name='Returns Distribution',
        showlegend=False
    ))
    
    fig_dist.update_layout(
        height=400,
        template='plotly_white',
        xaxis_title='Daily Returns (%)',
        yaxis_title='Frequency',
        bargap=0.1
    )
    
    st.plotly_chart(fig_dist, use_container_width=True)

# Footer with additional information
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.markdown("### Data Source")
    st.markdown("Clarksons Research - Shipping Intelligence Network")
with col2:
    st.markdown("### Notes")
    st.markdown("• All rates are in US dollars per day \n • Moving averages use weekly data \n• Volatility is calculated using rolling standard deviation")