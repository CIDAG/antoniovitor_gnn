from multiprocessing import connection


from pymongo import MongoClient

class Database():
    client = None
    connection = None
    connection_string = 'mongodb+srv://ic-user:cJsLbgvPGsRquWgP@cluster0.sor8o.gcp.mongodb.net/?retryWrites=true&w=majority'

    database_name = 'training_stats'
    instance = None

    @staticmethod
    def connect() -> None:
        db = Database
        if (db.connection is None):
            db.client = MongoClient(db.connection_string)
            db.instance = db.client[db.database_name]

    @staticmethod
    def getInstance():
        Database.connect()
        return Database.instance


class Collection():
    database = None

    @staticmethod
    def get(collection_name) -> None:
        if(Collection.database is None):
            Collection.database = Database.getInstance()

        return Collection.database[collection_name]




