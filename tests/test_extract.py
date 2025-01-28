from data_loader import OpenSkyLoader

loader = OpenSkyLoader()
sampled_data = loader.sample_flights_with_vectors(
    start_time='2024-03-01 08:00:00',
    end_time='2024-03-01 09:00:00',
    n_samples=5,
    airport='KATL'
)

# Access the sampled flights
flights = sampled_data['flights']
print(flights)

# Access their corresponding state vectors
vectors = sampled_data['state_vectors']
print(vectors)