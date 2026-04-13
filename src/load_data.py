import pandas as pd

# Load datasets
flights = pd.read_csv("data/flights.csv", low_memory=False)
airlines = pd.read_csv("data/airlines.csv")
airports = pd.read_csv("data/airports.csv")

# Reduction
flights = flights.sample(n=100000, random_state=42)

# Preview
print("Flights data:")
print(flights.head())

print("\nFlights info:")
print(flights.info())

print("\nAirlines data:")
print(airlines.head())

print("\nAirports data:")
print(airports.head())

# Analysis

# Top airlines
print("\nTop airlines by number of flights:")
print(flights["AIRLINE"].value_counts().head(10))

# Top origin airports
print("\nTop origin airports:")
print(flights["ORIGIN_AIRPORT"].value_counts().head(10))

# Average departure delay
print("\nAverage departure delay:")
print(flights["DEPARTURE_DELAY"].mean())

