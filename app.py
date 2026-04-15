import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Flight Delay Dashboard", layout="wide")

st.title("✈️ Flight Delay Analysis Dashboard")

@st.cache_data
def load_data():
    flights = pd.read_csv("data/flights.csv", low_memory=False)
    airlines = pd.read_csv("data/airlines.csv")

    flights = flights.sample(n=100000, random_state=42)

    flights = flights.merge(
        airlines,
        left_on="AIRLINE",
        right_on="IATA_CODE",
        how="left"
    )

    flights = flights.rename(columns={"AIRLINE_y": "AIRLINE_NAME"})

    return flights

# Load data once
flights = load_data()

# -------------------------------
# Sidebar filters
# -------------------------------
st.sidebar.header("Filters")

airline_options = ["All"] + sorted(flights["AIRLINE_NAME"].dropna().unique().tolist())
selected_airline_filter = st.sidebar.selectbox(
    "Choose an airline",
    airline_options
)

month_options = ["All"] + list(range(1, 13))
selected_month = st.sidebar.selectbox(
    "Select Month",
    month_options
)

# Apply filters
dashboard_flights = flights.copy()

if selected_airline_filter != "All":
    dashboard_flights = dashboard_flights[
        dashboard_flights["AIRLINE_NAME"] == selected_airline_filter
    ]

if selected_month != "All":
    dashboard_flights = dashboard_flights[
        dashboard_flights["MONTH"] == selected_month
    ]

# -------------------------------
# Overview metrics
# -------------------------------
st.subheader("Overview")

total_flights = len(dashboard_flights)
num_airlines = dashboard_flights["AIRLINE_NAME"].nunique()
avg_delay = dashboard_flights["DEPARTURE_DELAY"].mean()
median_delay = dashboard_flights["DEPARTURE_DELAY"].median()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Flights", f"{total_flights:,}")
col2.metric("Airlines", num_airlines)
col3.metric("Avg Departure Delay", f"{avg_delay:.2f} min" if pd.notna(avg_delay) else "N/A")
col4.metric("Median Departure Delay", f"{median_delay:.2f} min" if pd.notna(median_delay) else "N/A")

# -------------------------------
# Key insights
# -------------------------------
st.subheader("Key Insights")

st.markdown("""
- ✈️ Spirit Airlines tends to show the highest consistent delays
- 🟢 Hawaiian Airlines is often among the most punctual
- 📉 Most flights are close to on-time, with median delay near 0 minutes
- ⚠️ Average delays are affected by a small number of extreme outliers
""")

# -------------------------------
# Filtered airline comparison
# -------------------------------
filtered_flights = dashboard_flights[
    (dashboard_flights["DEPARTURE_DELAY"] > -20) &
    (dashboard_flights["DEPARTURE_DELAY"] < 120)
]

delay_by_airline = (
    filtered_flights.groupby("AIRLINE_NAME")["DEPARTURE_DELAY"]
    .mean()
    .sort_values(ascending=False)
)

# -------------------------------
# Top delayed airports
# -------------------------------
airport_counts = dashboard_flights["ORIGIN_AIRPORT"].value_counts()
valid_airports = airport_counts[airport_counts > 200].index

filtered_airports = dashboard_flights[
    dashboard_flights["ORIGIN_AIRPORT"].isin(valid_airports)
]

# Fall back if the strict airport threshold removes everything
if filtered_airports.empty:
    airport_source = dashboard_flights.copy()
else:
    airport_source = filtered_airports.copy()

# Remove rows where delay is missing
airport_source = airport_source.dropna(subset=["DEPARTURE_DELAY"])

# Build grouped result only if data remains
if airport_source.empty:
    airport_delay = pd.Series(dtype=float)
else:
    airport_delay = (
        airport_source.groupby("ORIGIN_AIRPORT")["DEPARTURE_DELAY"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

# -------------------------------
# Side-by-side charts
# -------------------------------
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Average Delay by Airline (Filtered)")

    if delay_by_airline.empty:
        st.info("No airline delay data available for the selected filters.")
    else:
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        delay_by_airline.plot(kind="bar", ax=ax1)
        ax1.set_title("Average Delay by Airline (Filtered)")
        ax1.set_xlabel("Airline")
        ax1.set_ylabel("Delay (minutes)")
        plt.xticks(rotation=45)
        st.pyplot(fig1)

with col_right:
    st.subheader("Top 10 Most Delayed Origin Airports")

    if airport_delay.empty:
        st.info("No airport delay data available for the selected filters.")
    else:
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        airport_delay.plot(kind="bar", ax=ax2)
        ax2.set_title("Top 10 Most Delayed Origin Airports")
        ax2.set_xlabel("Airport")
        ax2.set_ylabel("Average Delay (minutes)")
        plt.xticks(rotation=45)
        st.pyplot(fig2)
        
# -------------------------------
# Monthly delay trend
# -------------------------------
st.subheader("Average Departure Delay by Month")

if selected_month != "All":
    st.info("Monthly trend is most useful when viewing all months.")
else:
    monthly_delay = (
        dashboard_flights.groupby("MONTH")["DEPARTURE_DELAY"]
        .mean()
        .sort_index()
    )

    month_names = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
        5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
        9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
    }

    if monthly_delay.empty:
        st.info("No monthly delay data available for the selected filters.")
    else:
        monthly_delay.index = monthly_delay.index.map(month_names)

        fig3, ax3 = plt.subplots(figsize=(8, 4))
        monthly_delay.plot(kind="line", marker="o", ax=ax3)
        ax3.set_title("Average Departure Delay by Month")
        ax3.set_xlabel("Month")
        ax3.set_ylabel("Average Delay (minutes)")
        st.pyplot(fig3)

# -------------------------------
# Top origin airports
# -------------------------------
st.subheader("Top 10 Origin Airports")

top_origins = dashboard_flights["ORIGIN_AIRPORT"].value_counts().head(10)

if top_origins.empty:
    st.info("No origin airport data available for the selected filters.")
else:
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    top_origins.plot(kind="bar", ax=ax4)
    ax4.set_title("Top 10 Origin Airports")
    ax4.set_xlabel("Airport")
    ax4.set_ylabel("Number of Flights")
    plt.xticks(rotation=45)
    st.pyplot(fig4)

# -------------------------------
# Interactive airline explorer
# -------------------------------
st.subheader("Explore Delays by Airline")

explorer_airline_list = sorted(dashboard_flights["AIRLINE_NAME"].dropna().unique())

if explorer_airline_list:
    selected_airline = st.selectbox(
        "Select an airline for delay distribution:",
        explorer_airline_list
    )

    airline_data = dashboard_flights[
        dashboard_flights["AIRLINE_NAME"] == selected_airline
    ]

    departure_delays = airline_data["DEPARTURE_DELAY"].dropna()

    if departure_delays.empty:
        st.info("No delay distribution data available for this airline and filter combination.")
    else:
        fig5, ax5 = plt.subplots(figsize=(8, 4))
        departure_delays.plot(
            kind="hist",
            bins=50,
            ax=ax5
        )
        ax5.set_title(f"Delay Distribution for {selected_airline}")
        ax5.set_xlabel("Delay (minutes)")
        ax5.set_ylabel("Number of Flights")
        st.pyplot(fig5)
else:
    st.warning("No airline data available for the selected filters.")