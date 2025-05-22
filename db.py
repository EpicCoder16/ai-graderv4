import gspread
from oauth2client.service_account import ServiceAccountCredentials
import logging
from datetime import datetime
import bcrypt
import os
import json

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup Google Sheets credentials
SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
SPREADSHEET_NAME = "AI_Grader_Storage"  # Rename to your sheet name

# Load credentials from environment variable
google_creds_json = os.getenv('GOOGLE_CREDENTIALS')

if google_creds_json is None:
    raise ValueError("Missing GOOGLE_CREDENTIALS environment variable!")

# Parse the JSON string into a dictionary
google_creds_dict = json.loads(google_creds_json)

# Authorize gspread
credentials = ServiceAccountCredentials.from_json_keyfile_dict(google_creds_dict, SCOPE)
client = gspread.authorize(credentials)
sheet = client.open(SPREADSHEET_NAME)
users_ws = sheet.worksheet("users")
comparisons_ws = sheet.worksheet("comparisons")

# ------------------ USERS ------------------

def find_user(username):
    users = users_ws.get_all_records()
    for user in users:
        if user["username"] == username:
            return user
    return None

def register_user(username, password):
    if find_user(username):
        raise ValueError("Username already exists")

    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    new_id = len(users_ws.get_all_values())  # Simple ID logic (assumes header exists)
    users_ws.append_row([new_id, username, hashed])
    logger.info(f"User {username} registered successfully")

def login_user(username, password):
    user = find_user(username)
    if not user:
        raise ValueError("Invalid username or password")

    if not bcrypt.checkpw(password.encode('utf-8'), user["password_hash"].encode('utf-8')):
        raise ValueError("Invalid username or password")

    return {"message": "Login successful", "user_id": user["id"]}

# ------------------ COMPARISONS ------------------

def store_comparison(user_id, filename, similarity_score):
    timestamp = datetime.now().isoformat()
    comparisons_ws.append_row([user_id, filename, similarity_score, timestamp])
    logger.info(f"Stored comparison for user {user_id}")

def get_comparisons(user_id):
    all_rows = comparisons_ws.get_all_records()
    return [row for row in all_rows if str(row["user_id"]) == str(user_id)]
