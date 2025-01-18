from pyopensky.trino import Trino

# Initialize the Trino connection
trino = Trino()

# Execute the query to list tables
query = "SHOW TABLES"
tables = trino.query(query)

# Display the list of tables
print(tables)

# Iterate through each table and display its features
for table in tables['Table']:
    query = f"DESCRIBE {table}"
    features = trino.query(query)
    print(f"\nFeatures of table '{table}':")
    print(features)

