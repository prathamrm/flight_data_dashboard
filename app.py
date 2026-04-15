import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Flight Delay Dashboard", layout="wide")

st.title("✈️ Flight Delay Analysis Dashboard")
st.caption("Interactive dashboard for exploring airline delays, airport patterns, and predicting departure delay.")

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance in miles using the Haversine formula."""
    R = 3958.8  # Earth radius in miles

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


@st.cache_data
def load_data():
    flights = pd.read_csv("data/flights.csv", low_memory=False)
    airlines = pd.read_csv("data/airlines.csv")
    airports = pd.read_csv("data/airports.csv")

    # Keep app responsive
    flights = flights.sample(n=100000, random_state=42)

    # Merge airline names
    flights = flights.merge(
        airlines,
        left_on="AIRLINE",
        right_on="IATA_CODE",
        how="left"
    )

    flights = flights.rename(columns={
        "AIRLINE_x": "AIRLINE_CODE",
        "AIRLINE_y": "AIRLINE_NAME"
    })

    # Merge readable origin airport names + coordinates
    origin_airports = airports[["IATA_CODE", "AIRPORT", "LATITUDE", "LONGITUDE"]].rename(columns={
        "IATA_CODE": "ORIGIN_AIRPORT",
        "AIRPORT": "ORIGIN_AIRPORT_NAME",
        "LATITUDE": "ORIGIN_LAT",
        "LONGITUDE": "ORIGIN_LON"
    })

    flights = flights.merge(
        origin_airports,
        on="ORIGIN_AIRPORT",
        how="left"
    )

    # Merge readable destination airport names + coordinates
    destination_airports = airports[["IATA_CODE", "AIRPORT", "LATITUDE", "LONGITUDE"]].rename(columns={
        "IATA_CODE": "DESTINATION_AIRPORT",
        "AIRPORT": "DESTINATION_AIRPORT_NAME",
        "LATITUDE": "DEST_LAT",
        "LONGITUDE": "DEST_LON"
    })

    flights = flights.merge(
        destination_airports,
        on="DESTINATION_AIRPORT",
        how="left"
    )

    # Friendly labels for UI dropdowns
    flights["ORIGIN_LABEL"] = (
        flights["ORIGIN_AIRPORT"].astype(str) + " — " +
        flights["ORIGIN_AIRPORT_NAME"].fillna("Unknown Airport")
    )

    flights["DESTINATION_LABEL"] = (
        flights["DESTINATION_AIRPORT"].astype(str) + " — " +
        flights["DESTINATION_AIRPORT_NAME"].fillna("Unknown Airport")
    )

    return flights


@st.cache_resource
def load_model():
    return joblib.load("models/delay_predictor.joblib")


# Load once
flights = load_data()
model = load_model()

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

if filtered_airports.empty:
    airport_source = dashboard_flights.copy()
else:
    airport_source = filtered_airports.copy()

airport_source = airport_source.dropna(subset=["DEPARTURE_DELAY"])

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

# -------------------------------
# Flight Delay Prediction Tool
# -------------------------------
st.markdown("---")
st.subheader("✈️ Predict Flight Delay")
st.write("Enter flight details to estimate departure delay.")

month_input = st.selectbox("Month", list(range(1, 13)), key="pred_month")
day_input = st.selectbox("Day of Week", list(range(1, 8)), key="pred_day")

airline_input = st.selectbox(
    "Airline",
    sorted(flights["AIRLINE_NAME"].dropna().unique()),
    key="pred_airline"
)

# Only show airports with known readable names
known_origin_labels = sorted(
    flights.loc[
        flights["ORIGIN_AIRPORT_NAME"].notna(),
        "ORIGIN_LABEL"
    ].dropna().unique()
)

known_dest_labels = sorted(
    flights.loc[
        flights["DESTINATION_AIRPORT_NAME"].notna(),
        "DESTINATION_LABEL"
    ].dropna().unique()
)

origin_label = st.selectbox(
    "Origin Airport",
    known_origin_labels,
    key="pred_origin"
)

dest_label = st.selectbox(
    "Destination Airport",
    known_dest_labels,
    key="pred_dest"
)

# Build code maps for model input
airline_code_map = (
    flights[["AIRLINE_NAME", "AIRLINE_CODE"]]
    .dropna()
    .drop_duplicates()
    .set_index("AIRLINE_NAME")["AIRLINE_CODE"]
    .to_dict()
)

origin_code_map = (
    flights[["ORIGIN_LABEL", "ORIGIN_AIRPORT", "ORIGIN_LAT", "ORIGIN_LON"]]
    .dropna()
    .drop_duplicates()
    .set_index("ORIGIN_LABEL")
)

dest_code_map = (
    flights[["DESTINATION_LABEL", "DESTINATION_AIRPORT", "DEST_LAT", "DEST_LON"]]
    .dropna()
    .drop_duplicates()
    .set_index("DESTINATION_LABEL")
)

airline_code_input = airline_code_map[airline_input]

origin_row = origin_code_map.loc[origin_label]
dest_row = dest_code_map.loc[dest_label]

origin_input = origin_row["ORIGIN_AIRPORT"]
dest_input = dest_row["DESTINATION_AIRPORT"]

distance_input = calculate_distance(
    origin_row["ORIGIN_LAT"],
    origin_row["ORIGIN_LON"],
    dest_row["DEST_LAT"],
    dest_row["DEST_LON"]
)

st.write(f"Estimated Distance: {distance_input:.0f} miles")

if st.button("Predict Delay"):
    input_df = pd.DataFrame([{
        "MONTH": month_input,
        "DAY_OF_WEEK": day_input,
        "AIRLINE": airline_code_input,
        "ORIGIN_AIRPORT": origin_input,
        "DESTINATION_AIRPORT": dest_input,
        "DISTANCE": distance_input
    }])

    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Departure Delay: {prediction:.2f} minutes")

st.markdown("---")
st.markdown("Built by Pratham | Flight Delay Analysis & Prediction Dashboard ✈️")