from flask import Blueprint, request, jsonify
from models import users_collection
import re
import bcrypt
import uuid

user_routes = Blueprint("user_routes", __name__)

name_regex = re.compile(r"^[A-Za-z .']+$")
email_regex = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")

@user_routes.route("/reg", methods=["POST"])
def register():
    try:
        data = request.json
        name = data.get("name")
        email = data.get("email")
        phoneno = data.get("phoneno")
        country = data.get("country")
        password = data.get("password")
        confirm_password = data.get("confirmPassword")

        print(name, email, phoneno, country, password, confirm_password)

        if not all([name, email, phoneno, country, password, confirm_password]):
            return jsonify({"message": "All fields are required"}), 400

        if not name_regex.match(name):
            return jsonify({
                "message": "Name should not contain special characters except (., ')"
            }), 400

        if not email_regex.match(email):
            return jsonify({"message": "Invalid email format"}), 400

        if password != confirm_password:
            return jsonify({"message": "Passwords do not match"}), 400

        existing_user = users_collection.find_one({"email": email})
        if existing_user:
            return jsonify({"message": "Email already registered"}), 400

        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        user_id = str(uuid.uuid4())

        users_collection.insert_one({
            "userId": user_id,
            "name": name,
            "email": email,
            "phoneno": phoneno,
            "country": country,
            "password": hashed_password
        })

        return jsonify({
            "success": True,
            "message": "User registered successfully"
        }), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@user_routes.route("/log", methods=["POST"])
def login():
    try:
        data = request.json
        email = data.get("email")
        password = data.get("password")

        if not email or not password:
            return jsonify({
                "message": "Email and password required"
            }), 400

        user = users_collection.find_one({"email": email})
        if not user:
            return jsonify({"message": "User not found"}), 400

        if not bcrypt.checkpw(password.encode('utf-8'), user["password"]):
            return jsonify({"message": "Invalid password"}), 400

        return jsonify({
            "success": True,
            "message": "Login successful"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@user_routes.route("/displaydata", methods=["POST"])
def display_data():
    try:
        data = request.json
        email = data.get("email")

        print("request received")

        if not email:
            return jsonify({"message": "Email required"}), 400

        user = users_collection.find_one({"email": email}, {"password": 0})

        if not user:
            return jsonify({"message": "User not found"}), 404

        user["_id"] = str(user["_id"])  # convert ObjectId to string

        return jsonify(user)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
