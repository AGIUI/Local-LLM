import lancedb
import pyarrow as pa
import json

embedding_models=[
    "",
    ""
]

class LanceDBAssistant:
    def __init__(self, dirpath, filename,n=384):
        self.dirpath = dirpath
        self.filename = filename
        self.db = None
        self.create_schema(n)

    def create_schema(self,n=384):
        self.schema = pa.schema([
            pa.field("vector", pa.list_(pa.float32(), n)),
            pa.field("item", pa.string()),
            pa.field("id", pa.string()),
        ])

    def connect(self):
        if self.db is None:
            self.db = lancedb.connect(self.dirpath)

    def create(self):
        self.connect()
        table = self.db.create_table(self.filename, schema=self.schema, mode="overwrite")
        return table
    
    def open(self):
        table=None
        try:
            ts=self.db.table_names()
            if self.filename in ts:
                table = self.db.open_table(self.filename)
        except:
            print('Creating a new table')
        return table

    def add(self, data):
        self.connect()
        table = self.open()
        
        if table is None:
            table = self.create()  # Assuming data is a pyarrow.Table

        table.add(data=data,
                #   mode="overwrite" //这个导致了bug，全部覆盖了
                  )

        return self.db[self.filename].head()

    def search(self, vector, limit=5):
        self.connect()
        table = self.open()
        res=[]
        if table:
            res = table.search(vector).select(['id','item']).limit(limit).to_list()
            res=[{
                'id':r['id'],
                'item':json.loads(r['item']),
                '_distance':r['_distance']
            } for r in res]
            return res

    def list_tables(self):
        self.connect()
        return self.db.table_names()

    def delete_table(self,filename):
        self.connect()
        return self.db.drop_table(filename, ignore_missing=True)

    def get_by_id(self,id):
        self.connect()
        table = self.open()
        if table:
            items=table.search().where(f"id = '{id}'", prefilter=True).select(['id']).to_list()
            for item in items:
                if item['id']==id:
                    return item
        return
    def update(self,id,item):
        self.connect()
        table = self.open()
        if table:
            table.update(where=f"id = '{id}'", values={"item":item})
             
# dirpath = "tmp/sample-lancedb"
# filename = "my_table2"

# assistant = LanceDBAssistant(dirpath, filename)

# # Create a new table
# assistant.create_schema()

# table = assistant.create(schema)

# # Add new data
# data =  [{"vector": [1.3, 1.4], "item": "fizz" },
#       {"vector": [9.5, 56.2], "item": "buzz" }]
# assistant.add(data)

# # Search by vector
# vector = [1.3, 1.4]  # Your search vector
# results = assistant.search(vector)

# # List all tables
# tables = assistant.list_tables()
# print(results)
# Delete the table
# assistant.delete_table()