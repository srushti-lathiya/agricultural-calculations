import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import urllib.parse
import urllib.request
from io import BytesIO
import math
import numpy as np
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv

# Set page configuration
st.set_page_config(layout="wide", page_title="Agricultural Calculations")

# Initialize session state for all inputs to track resets
if 'reset_clicked' not in st.session_state:
    st.session_state['reset_clicked'] = False

# Initialize all other session state variables if they don't exist
if 'city' not in st.session_state:
    st.session_state['city'] = ""
if 'state' not in st.session_state:
    st.session_state['state'] = ""
if 'start_date' not in st.session_state:
    st.session_state['start_date'] = None
if 'end_date' not in st.session_state:
    st.session_state['end_date'] = None
if 't_base' not in st.session_state:
    st.session_state['t_base'] = 0.0
if 'use_modified' not in st.session_state:
    st.session_state['use_modified'] = False
if 't_lower' not in st.session_state:
    st.session_state['t_lower'] = 0.0
if 't_upper' not in st.session_state:
    st.session_state['t_upper'] = 0.0
if 'harvest_date' not in st.session_state:
    st.session_state['harvest_date'] = None
if 'starting_moisture' not in st.session_state:
    st.session_state['starting_moisture'] = 0.80
if 'swath_density' not in st.session_state:
    st.session_state['swath_density'] = 450.0
if 'application_rate' not in st.session_state:
    st.session_state['application_rate'] = 0.0

# Inject custom CSS to hide the top bar
st.markdown(
    """
    <style>
    /* Hide the entire header bar */
    [data-testid="stHeader"] {
        display: none;
    }
    /* Hide the Streamlit watermark/footer */
    [data-testid="stDecoration"] {
        display: none;
    }
    /* Adjust padding if needed */
    [data-testid="stAppViewContainer"] {
        padding-top: 0px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def reset_inputs():
    """Reset all input values in session state"""
    st.session_state['city'] = ""
    st.session_state['state'] = ""
    st.session_state['start_date'] = None
    st.session_state['end_date'] = None
    st.session_state['t_base'] = 0.0
    st.session_state['use_modified'] = False
    st.session_state['t_lower'] = 0.0
    st.session_state['t_upper'] = 0.0
    st.session_state['harvest_date'] = None
    st.session_state['starting_moisture'] = 0.80
    st.session_state['swath_density'] = 450.0
    st.session_state['application_rate'] = 0.0
    st.session_state['reset_clicked'] = True

# =================================================================
# 1) Function to Fetch Weather Data from Visual Crossing
# =================================================================
def fetch_weather_data_from_api(api_key, city, state, start_date, end_date,
                                unit_group="us", elements="datetime,tempmin,tempmax"):
    """
    Fetches weather data from Visual Crossing between start_date and end_date
    for a given city, state (e.g. city='Ithaca', state='NY').
    Returns a DataFrame with columns ['datetime','tempmin','tempmax',...].
    """
    location_string = f"{city}, {state}"
    encoded_location = urllib.parse.quote(location_string)

    base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
    url = (
        f"{base_url}/{encoded_location}/{start_date}/{end_date}"
        f"?unitGroup={unit_group}&include=days&elements={elements}"
        f"&key={api_key}&contentType=csv"
    )

    try:
        response = urllib.request.urlopen(url)
        df_api = pd.read_csv(BytesIO(response.read()))
        return df_api
    except urllib.error.HTTPError as e:
        st.error(f"HTTP Error: {e.code} - {e.read().decode()}")
    except urllib.error.URLError as e:
        st.error(f"URL Error: {e.reason}")

    return pd.DataFrame()  # Return empty if error

def fetch_drying_weather_data_from_api(api_key, city, state, start_date, end_date,
                                unit_group="us"):
    """
    Fetches daily and hourly weather data for drying calculations.
    """
    location_string = f"{city}, {state}"
    encoded_location = urllib.parse.quote(location_string)

    base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"

    # Daily data (temperature, dew point, soil moisture)
    daily_url = (
        f"{base_url}/{encoded_location}/{start_date}/{end_date}"
        f"?unitGroup={unit_group}&include=days&elements=datetime,temp,dew,soilmoisturevol01"
        f"&key={api_key}&contentType=csv"
    )

    try:
        response = urllib.request.urlopen(daily_url)
        df_days = pd.read_csv(BytesIO(response.read()))
    except urllib.error.HTTPError as e:
        st.error(f"HTTP Error: {e.code} - {e.read().decode()}")
        return pd.DataFrame(), pd.DataFrame()
    except urllib.error.URLError as e:
        st.error(f"URL Error: {e.reason}")
        return pd.DataFrame(), pd.DataFrame()

    # Hourly data (solar radiation)
    hourly_url = (
        f"{base_url}/{encoded_location}/{start_date}/{end_date}"
        f"?unitGroup={unit_group}&include=hours&elements=datetime,solarradiation"
        f"&key={api_key}&contentType=csv"
    )

    try:
        response = urllib.request.urlopen(hourly_url)
        df_hours = pd.read_csv(BytesIO(response.read()))
    except urllib.error.HTTPError as e:
        st.error(f"HTTP Error: {e.code} - {e.read().decode()}")
        return df_days, pd.DataFrame()
    except urllib.error.URLError as e:
        st.error(f"URL Error: {e.reason}")
        return df_days, pd.DataFrame()

    return df_days, df_hours

# =================================================================
# 2) GDD Calculation Functions
# =================================================================
def calc_average_gdd(t_min, t_max, T_base):
    return max(((t_max + t_min) / 2) - T_base, 0)

def calc_single_sine_gdd(t_min, t_max, T_base):
    T_mean = (t_max + t_min) / 2
    A = (t_max - t_min) / 2
    if t_max <= T_base:
        return 0.0
    if t_min >= T_base:
        return (T_mean - T_base)
    alpha = (T_base - T_mean) / A
    theta = math.acos(alpha)
    dd = ((T_mean - T_base)*(math.pi - 2*theta) + A*math.sin(2*theta)) / math.pi
    return dd

def calc_single_triangle_gdd(t_min, t_max, T_base):
    if t_max <= T_base:
        return 0.0
    if t_min >= T_base:
        return ((t_max + t_min)/2 - T_base)
    proportion_of_day = (t_max - T_base) / (t_max - t_min)
    avg_above = ((t_max + T_base) / 2) - T_base
    dd = proportion_of_day * avg_above
    return max(dd, 0)

def calc_double_sine_gdd(t_min_today, t_max_today, t_min_tomorrow, T_base):
    seg1 = calc_single_sine_gdd(t_min_today, t_max_today, T_base) * 0.5
    seg2 = calc_single_sine_gdd(t_min_tomorrow, t_max_today, T_base) * 0.5
    return seg1 + seg2

def calc_double_triangle_gdd(t_min_today, t_max_today, t_min_tomorrow, T_base):
    seg1 = calc_single_triangle_gdd(t_min_today, t_max_today, T_base) * 0.5
    seg2 = calc_single_triangle_gdd(t_min_tomorrow, t_max_today, T_base) * 0.5
    return seg1 + seg2

def calculate_daily_gdd(row, df,
                        method="average",
                        T_base=50.0,
                        use_modified=False,
                        T_lower=50.0,
                        T_upper=86.0):
    idx = row.name
    t_max = row['tmax']
    t_min = row['tmin']
    if use_modified:
        t_max = min(t_max, T_upper)
        t_min = max(t_min, T_lower)
    if method == "average":
        return calc_average_gdd(t_min, t_max, T_base)
    elif method == "sine":
        return calc_single_sine_gdd(t_min, t_max, T_base)
    elif method == "triangle":
        return calc_single_triangle_gdd(t_min, t_max, T_base)
    elif method == "double_sine":
        if idx < len(df) - 1:
            t_min_next = df.loc[idx+1, 'tmin']
            if use_modified:
                t_min_next = max(t_min_next, T_lower)
            return calc_double_sine_gdd(t_min, t_max, t_min_next, T_base)
        else:
            return calc_single_sine_gdd(t_min, t_max, T_base)
    elif method == "double_triangle":
        if idx < len(df) - 1:
            t_min_next = df.loc[idx+1, 'tmin']
            if use_modified:
                t_min_next = max(t_min_next, T_lower)
            return calc_double_triangle_gdd(t_min, t_max, t_min_next, T_base)
        else:
            return calc_single_triangle_gdd(t_min, t_max, T_base)
    else:
        raise ValueError(f"Unknown method: {method}")

def calculate_gdds_for_df(df,
                          start_date,
                          end_date,
                          method="average",
                          T_base=50.0,
                          use_modified=False,
                          T_lower=50.0,
                          T_upper=86.0):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    df['daily_gdd'] = df.apply(
        lambda row: calculate_daily_gdd(row, df,
                                        method=method,
                                        T_base=T_base,
                                        use_modified=use_modified,
                                        T_lower=T_lower,
                                        T_upper=T_upper),
        axis=1
    )
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    df.loc[df['date'] < start_date, 'daily_gdd'] = 0
    df.loc[df['date'] > end_date, 'daily_gdd'] = 0
    df['cumulative_gdd'] = df['daily_gdd'].cumsum()
    return df

# =================================================================
# 3) Drying Calculation Functions 
# =================================================================
def merge_dfs(daily_df, hourly_df):
    '''
    Extracts the peak solar radiation for each day from the
    hourly df and adds it to the daily df.
    '''
    d_df = daily_df
    h_df = hourly_df

    # Step 1: Convert datetime columns to datetime objects
    h_df['datetime'] = pd.to_datetime(h_df['datetime'])
    d_df['datetime'] = pd.to_datetime(d_df['datetime'])

    # Step 2: Extract date from hourly timestamps
    h_df['date'] = h_df['datetime'].dt.date

    # Step 3: Group by date and get max solar radiation
    daily_peaks = h_df.groupby('date')['solarradiation'].max().reset_index()
    daily_peaks.rename(columns={'solarradiation': 'peak_solarradiation'}, inplace=True)

    # Step 4: Merge with daily dataframe
    # Convert daily_df datetime to just date for merging
    d_df['date'] = d_df['datetime'].dt.date
    merged_df = pd.merge(d_df, daily_peaks, on='date', how='left')

    # (Optional) Drop the 'date' column if not needed
    merged_df.drop(columns='date', inplace=True)

    return merged_df

def calculate_vapor_pressure_deficit(df):
    '''Takes weather df (with temp and dew in °F) and adds vapor pressure deficit column in kPa'''

    # Convert temperature and dew point from Fahrenheit to Celsius
    df['temp_C'] = (df['temp'] - 32) / 1.8
    df['dew_C'] = (df['dew'] - 32) / 1.8

    # Calculate saturation and actual vapor pressure
    df['saturation_vapor_pressure'] = 0.6108 * np.exp((17.27 * df['temp_C']) / (df['temp_C'] + 237.3))
    df['actual_vapor_pressure'] = 0.6108 * np.exp((17.27 * df['dew_C']) / (df['dew_C'] + 237.3))

    # Calculate VPD
    df['vapor_pressure_deficit'] = df['saturation_vapor_pressure'] - df['actual_vapor_pressure']

    return df

def swath_density_conversion(plants_per_sqft, g_per_plant=25):
    '''convert swath density from plants/ft^2 to g/m^2'''
    plants_per_sqm = plants_per_sqft * 10.764
    return plants_per_sqm * g_per_plant  # returns g/m²

def calculate_drying_rate_constant(SI, VPD, DAY, SM, SD, AR=0):
    '''
    SI = solar insolation, W/m^2
    VPD = vapor pressure deficit, kPA
    DAY = 1 for first day, 0 otherwise
    SM = soil moisture content, % dry basis
    SD = swath density, g/m^2
    AR = application rate of chemical solution, g_solution/g_dry-matter
    '''
    drying_rate = ((SI * (1. + 9.03*AR)) + (43.8 * VPD)) / ((61.4 * SM) + SD * (1.82 - 0.83 * DAY) * ((1.68 + 24.8 * AR)) + 2767)
    return drying_rate

def predict_moisture_content(df, startdate, swath_density=450, starting_moisture=0.80, application_rate=0):
    """
    Simulate daily drying and return DataFrame with moisture content predictions.
    """
    # Ensure datetime and sort
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df[df['datetime'] >= pd.to_datetime(startdate)].copy()
    df.sort_values('datetime', inplace=True)

    # Calculate VPD
    df = calculate_vapor_pressure_deficit(df)

    # Initialize columns
    moisture_contents = [starting_moisture]
    drying_rates = []
    current_moisture = starting_moisture

    for idx, row in df.iterrows():
        day_number = len(moisture_contents) - 1
        DAY = 1 if day_number == 0 else 0

        SI = row['peak_solarradiation']
        VPD = row['vapor_pressure_deficit']
        SM = 100 * row['soilmoisturevol01'] if not np.isnan(row['soilmoisturevol01']) else 10  # Default if NaN
        SD = swath_density
        AR = application_rate

        k = calculate_drying_rate_constant(SI, VPD, DAY, SM, SD, AR)

        # Update moisture content
        current_moisture *= np.exp(-k)
        moisture_contents.append(current_moisture)
        drying_rates.append(k)

        # Stop if moisture is below 0.08 (8%)
        if current_moisture <= 0.08:
            break

    # Create result DataFrame
    result_df = df.iloc[:len(moisture_contents)-1].copy()
    result_df['drying_rates'] = drying_rates
    result_df['predicted_moisture'] = moisture_contents[:-1]
    # Convert moisture to percentage for display
    result_df['predicted_moisture_pct'] = result_df['predicted_moisture'] * 100

    result_df = result_df.dropna(subset=['predicted_moisture'])

    return result_df

# =================================================================
# 4) Main Application
# =================================================================
def main():
    load_dotenv()  # Load environment variables from .env file
    API_KEY = os.getenv("API_KEY")  # Replace with your Visual Crossing API key

    # Remove the app title since it might be contributing to the "undefined" issue
    # Instead, use a container with custom styling
    title_container = st.container()
    with title_container:
        st.markdown("<h1 style='text-align: center;'>zalliant-Integrating Forecasting for Weather-Optimized Crop Cutting</h1>", unsafe_allow_html=True)

    # Create sidebar for all inputs
    with st.sidebar:
        st.write("### Input Parameters")

        # Algorithm Selection - make non-editable
        algorithm_options = ["GDD Calculation", "Drying Calculation"]
        algorithm = st.selectbox(
            "Select Calculation Algorithm",
            options=algorithm_options,
            key="algorithm"
        )

        # GDD Calculation inputs
        if algorithm == "GDD Calculation":
            st.write("### GDD Parameters")

            # Location inputs
            city = st.text_input("City", value=st.session_state['city'])
            state = st.text_input("State", value=st.session_state['state'])

            # Custom date inputs without defaults
            start_date = st.date_input("Start Date", value=st.session_state['start_date'] if st.session_state['start_date'] else None)
            end_date = st.date_input("End Date", value=st.session_state['end_date'] if st.session_state['end_date'] else None)

            # Method selection - make non-editable
            method_options = ["average", "sine", "triangle", "double_sine", "double_triangle"]
            method = st.selectbox(
                "Method",
                options=method_options
            )

            # Base temperature
            t_base = st.number_input("T_base", value=st.session_state['t_base'], step=0.1)

            # Modified GDD option
            use_modified = st.checkbox("Use Modified?", value=st.session_state['use_modified'])

            # Only show T_lower and T_upper if use_modified is checked
            if use_modified:
                t_lower = st.number_input("T_lower", value=st.session_state['t_lower'], step=0.1)
                t_upper = st.number_input("T_upper", value=st.session_state['t_upper'], step=0.1)
            else:
                t_lower = 50.0
                t_upper = 86.0

            # Save input values to session state
            st.session_state['city'] = city
            st.session_state['state'] = state
            st.session_state['start_date'] = start_date
            st.session_state['end_date'] = end_date
            st.session_state['t_base'] = t_base
            st.session_state['use_modified'] = use_modified
            st.session_state['t_lower'] = t_lower
            st.session_state['t_upper'] = t_upper

            # Calculate button
            calculate_button = st.button("Get Data & Calculate")

        # Drying Calculation inputs
        elif algorithm == "Drying Calculation":
            st.write("### Drying Parameters")

            # Main inputs from paste-2.txt - now showing all parameters directly (no expander)
            city = st.text_input("City", value=st.session_state['city'])
            state = st.text_input("State", value=st.session_state['state'] )
            harvest_date = st.date_input("Harvest Date", value=st.session_state['harvest_date'])



            # Save input values to session state
            st.session_state['city'] = city
            st.session_state['state'] = state
            st.session_state['harvest_date'] = harvest_date


            # Calculate button
            drying_button = st.button("Estimate Drying")

        # Reset button at the bottom of sidebar
        st.markdown("---")
        if st.button("Reset Inputs"):
            reset_inputs()
            st.rerun()

    # Main content area - for results only
    if algorithm == "GDD Calculation" and calculate_button:
        # Validate inputs
        if not city or not state:
            st.error("Please enter both City and State")
        elif start_date is None or end_date is None:
            st.error("Please select both Start Date and End Date")
        elif start_date >= end_date:
            st.error("End Date must be after Start Date")
        else:
            # Show loading message
            with st.spinner("Fetching weather data and calculating GDD..."):
                # Fetch data
                df_api = fetch_weather_data_from_api(
                    api_key=API_KEY,
                    city=city,
                    state=state,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                    unit_group="us",
                    elements="datetime,tempmin,tempmax"
                )

                if not df_api.empty:
                    # Rename columns
                    df_renamed = df_api.copy()
                    df_renamed.rename(columns={
                        "datetime": "date",
                        "tempmax": "tmax",
                        "tempmin": "tmin"
                    }, inplace=True)

                    # Calculate GDD
                    df_calc = calculate_gdds_for_df(
                        df=df_renamed,
                        start_date=start_date,
                        end_date=end_date,
                        method=method,
                        T_base=t_base,
                        use_modified=use_modified,
                        T_lower=t_lower,
                        T_upper=t_upper
                    )

                    # Create plot
                    location_str = f"{city}, {state}"
                    st.header(f"Cumulative GDD Results")
                    st.subheader(f"Location: {location_str} | Method: {method}")

                    fig = px.line(
                        df_calc,
                        x="date",
                        y="cumulative_gdd",
                        title=f"Cumulative GDD Chart"
                    )
                    fig.update_layout(
                        xaxis_title="Date", 
                        yaxis_title="Cumulative GDD",
                        height=600
                    )

                    # Display plot in main area
                    st.plotly_chart(fig, use_container_width=True)

                else:
                    st.error("Failed to fetch weather data. Please check your inputs and try again.")

    elif algorithm == "Drying Calculation" and drying_button:
        # Validate inputs
        if not city or not state:
            st.error("Please enter both City and State")
        elif harvest_date is None:
            st.error("Please select a Harvest Date")
        else:
            # Show loading message
            with st.spinner("Fetching weather data and estimating drying..."):
                # Calculate forecast period (2 months from harvest date)
                harvest_date_str = harvest_date.strftime("%Y-%m-%d")
                end_date_dt = harvest_date + relativedelta(months=2)
                end_date_str = end_date_dt.strftime("%Y-%m-%d")

                # Fetch data
                daily_df, hourly_df = fetch_drying_weather_data_from_api(
                    api_key=API_KEY,
                    city=city,
                    state=state,
                    start_date=harvest_date_str,
                    end_date=end_date_str,
                    unit_group="us"
                )

                if not daily_df.empty and not hourly_df.empty:
                    # Merge dataframes
                    merged_df = merge_dfs(daily_df, hourly_df)

                    # Calculate drying prediction
                    drying_df = predict_moisture_content(
                        df=merged_df,
                        startdate=harvest_date_str,

                    )

                    # Display results
                    location_str = f"{city}, {state}"
                    st.header(f"Crop Drying Prediction Results")
                    st.subheader(f"Location: {location_str} | Harvest Date: {harvest_date_str}")

                    # Create plot
                    fig = px.line(
                        drying_df,
                        x="datetime",
                        y="predicted_moisture_pct",
                        title=f"Expected Crop Moisture Content Over Time"
                    )
                    fig.update_layout(
                        xaxis_title="Date", 
                        yaxis_title="Moisture Content (%)",
                        height=600
                    )

                    # Add target line for baling moisture content (15%)
                    fig.add_shape(
                        type="line",
                        x0=drying_df["datetime"].min(),
                        y0=15,
                        x1=drying_df["datetime"].max(),
                        y1=15,
                        line=dict(
                            color="green",
                            width=2,
                            dash="dash",
                        )
                    )

                    # Display plot in main area
                    st.plotly_chart(fig, use_container_width=True)

                    # Calculate days to reach 15% moisture (good for baling)
                    target_moisture = 0.15
                    baling_ready_row = drying_df[drying_df['predicted_moisture'] <= target_moisture].iloc[0] if any(drying_df['predicted_moisture'] <= target_moisture) else None

                    if baling_ready_row is not None:
                        baling_date = pd.to_datetime(baling_ready_row['datetime'])
                        days_to_baling = (baling_date - pd.to_datetime(harvest_date_str)).days

                        # Create info box for baling date
                        st.info(f"Based on weather forecasts, the crop will reach optimal baling moisture (15%) in approximately **{days_to_baling} days** from harvest (around {baling_date.strftime('%B %d, %Y')}).")
                    else:
                        st.warning("The crop may not reach optimal baling moisture (15%) within the forecast period.")

                else:
                    st.error("Failed to fetch weather data. Please check your inputs and try again.")

if __name__ == "__main__":
    main()