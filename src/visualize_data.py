import pandas as pd
import matplotlib.pyplot as plt

# Load datasets
flights = pd.read_csv("data/flights.csv", low_memory=False)
airlines = pd.read_csv("data/airlines.csv")

# Sample
flights = flights.sample(n=100000, random_state=42)

# Merge flights with airline names
flights = flights.merge(
    airlines,
    left_on="AIRLINE",
    right_on="IATA_CODE",
    how="left"
)

# Rename column for clarity
flights = flights.rename(columns={"AIRLINE_y": "AIRLINE_NAME"})

# -------------------------------
# Top airlines by number of flights
# -------------------------------
top_airlines = flights["AIRLINE_NAME"].value_counts().head(10)

plt.figure(figsize=(12, 6))
top_airlines.plot(kind="bar")

plt.title("Top 10 Airlines by Number of Flights")
plt.xlabel("Airline")
plt.ylabel("Number of Flights")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# -------------------------------
# Average departure delay by airline
# -------------------------------
delay_by_airline = (
    flights.groupby("AIRLINE_NAME")["DEPARTURE_DELAY"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(12, 6))
delay_by_airline.plot(kind="bar")

plt.title("Top 10 Airlines by Average Departure Delay")
plt.xlabel("Airline")
plt.ylabel("Average Delay (minutes)")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# -------------------------------
# Distribution of delays
# -------------------------------
plt.figure(figsize=(10, 6))
flights["DEPARTURE_DELAY"].dropna().plot(kind="hist", bins=50)

plt.title("Distribution of Departure Delays")
plt.xlabel("Delay (minutes)")
plt.ylabel("Number of Flights")

plt.tight_layout()
plt.show()

# -------------------------------
# Median delay by airline
# -------------------------------
median_delay = (
    flights.groupby("AIRLINE_NAME")["DEPARTURE_DELAY"]
    .median()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(12, 6))
median_delay.plot(kind="bar")

plt.title("Top 10 Airlines by Median Departure Delay")
plt.xlabel("Airline")
plt.ylabel("Median Delay (minutes)")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# -------------------------------
# Filtered delay distribution
# -------------------------------
filtered_delays = flights["DEPARTURE_DELAY"].dropna()
filtered_delays = filtered_delays[
    (filtered_delays > -50) & (filtered_delays < 200)
]

plt.figure(figsize=(10, 6))
filtered_delays.plot(kind="hist", bins=50)

plt.title("Filtered Distribution of Departure Delays")
plt.xlabel("Delay (minutes)")
plt.ylabel("Number of Flights")

plt.tight_layout()
plt.show()

# -------------------------------
# Filtered average delay by airline (better comparison)
# -------------------------------
filtered_flights = flights[
    (flights["DEPARTURE_DELAY"] > -20) &
    (flights["DEPARTURE_DELAY"] < 120)
]

delay_by_airline_filtered = (
    filtered_flights.groupby("AIRLINE_NAME")["DEPARTURE_DELAY"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(12, 6))
delay_by_airline_filtered.plot(kind="bar")

plt.title("Average Departure Delay by Airline (Filtered)")
plt.xlabel("Airline")
plt.ylabel("Average Delay (minutes)")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# -------------------------------
# Best vs Worst Airlines (Filtered)
# -------------------------------
sorted_delays = delay_by_airline_filtered

worst_airlines = sorted_delays.head(5)
best_airlines = sorted_delays.tail(5)

plt.figure(figsize=(12, 6))

# Combine for plotting
combined = pd.concat([worst_airlines, best_airlines])

combined.plot(kind="bar")

plt.title("Best and Worst Airlines by Average Delay (Filtered)")
plt.xlabel("Airline")
plt.ylabel("Average Delay (minutes)")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()