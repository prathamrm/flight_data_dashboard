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

# Now use full airline names
top_airlines = flights["AIRLINE_y"].value_counts().head(10)

# Plot
plt.figure(figsize=(12, 6))
top_airlines.plot(kind="bar")

plt.title("Top 10 Airlines by Number of Flights")
plt.xlabel("Airline")
plt.ylabel("Number of Flights")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Calculate average departure delay by airline
delay_by_airline = flights.groupby("AIRLINE_y")["DEPARTURE_DELAY"].mean().sort_values(ascending=False).head(10)

# Plot
plt.figure(figsize=(12, 6))
delay_by_airline.plot(kind="bar")

plt.title("Top 10 Airlines by Average Departure Delay")
plt.xlabel("Airline")
plt.ylabel("Average Delay (minutes)")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()