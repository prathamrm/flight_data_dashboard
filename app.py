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

# Load once
flights = load_data()

# -------------------------------
# Summary metrics (TOP)
# -------------------------------
st.subheader("Overview")

total_flights = len(flights)
num_airlines = flights["AIRLINE_NAME"].nunique()
avg_delay = flights["DEPARTURE_DELAY"].mean()
median_delay = flights["DEPARTURE_DELAY"].median()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Flights", f"{total_flights:,}")
col2.metric("Airlines", num_airlines)
col3.metric("Avg Departure Delay", f"{avg_delay:.2f} min")
col4.metric("Median Departure Delay", f"{median_delay:.2f} min")

# -------------------------------
# Filtered airline comparison
# -------------------------------
filtered_flights = flights[
    (flights["DEPARTURE_DELAY"] > -20) &
    (flights["DEPARTURE_DELAY"] < 120)
]

delay_by_airline = (
    filtered_flights.groupby("AIRLINE_NAME")["DEPARTURE_DELAY"]
    .mean()
    .sort_values(ascending=False)
)

st.subheader("Average Delay by Airline (Filtered)")

fig, ax = plt.subplots(figsize=(10, 5))
delay_by_airline.plot(kind="bar", ax=ax)

ax.set_title("Average Delay by Airline (Filtered)")
ax.set_xlabel("Airline")
ax.set_ylabel("Delay (minutes)")
plt.xticks(rotation=45)

st.pyplot(fig)

# -------------------------------
# Airport traffic
# -------------------------------
st.subheader("Top 10 Origin Airports")

top_origins = flights["ORIGIN_AIRPORT"].value_counts().head(10)

fig3, ax3 = plt.subplots(figsize=(10, 5))
top_origins.plot(kind="bar", ax=ax3)

ax3.set_title("Top 10 Origin Airports")
ax3.set_xlabel("Airport")
ax3.set_ylabel("Number of Flights")
plt.xticks(rotation=45)

st.pyplot(fig3)

# -------------------------------
# Interactive airline explorer
# -------------------------------
st.subheader("Explore Delays by Airline")

airline_list = sorted(flights["AIRLINE_NAME"].dropna().unique())

selected_airline = st.selectbox(
    "Select an airline:",
    airline_list
)

airline_data = flights[
    flights["AIRLINE_NAME"] == selected_airline
]

fig2, ax2 = plt.subplots(figsize=(10, 5))

airline_data["DEPARTURE_DELAY"].dropna().plot(
    kind="hist",
    bins=50,
    ax=ax2
)

ax2.set_title(f"Delay Distribution for {selected_airline}")
ax2.set_xlabel("Delay (minutes)")
ax2.set_ylabel("Number of Flights")

st.pyplot(fig2)