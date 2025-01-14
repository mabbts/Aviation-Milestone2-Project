from pyopensky.trino import Trino

# Initialize the Trino connection
trino = Trino()

# Execute the query to list tables
query = "SHOW TABLES"
tables = trino.query(query)

# Display the list of tables
print(tables)