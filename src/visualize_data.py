import pandas as pd
import matplotlib.pyplot as plt

# Load data
flights = pd.read_csv("data/flights.csv", low_memory=False)

# Sample to keep it fast
flights = flights.sample(n=100000, random_state=42)

# Get top airlines
top_airlines = flights["AIRLINE"].value_counts().head(10)

# Plot
plt.figure(figsize=(10, 6))
top_airlines.plot(kind="bar")

plt.title("Top 10 Airlines by Number of Flights")
plt.xlabel("Airline Code")
plt.ylabel("Number of Flights")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()