from config import connect_db

db = connect_db()
users_collection = db["users"]