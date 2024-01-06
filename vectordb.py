import lancedb

uri = "data/sample-lancedb"
db = lancedb.connect(uri)
table = db.create_table("my_table",
                         data=[{"vector": [3.1, 4.1], "item": "foo", "price": 10.0},
                               {"vector": [5.9, 26.5], "item": "bar", "price": 20.0}])
result = table.search([100, 100]).limit(2).to_pandas()

print(result)