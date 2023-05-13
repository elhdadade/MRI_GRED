from fastapi import FastAPI, File, UploadFile
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from uuid import uuid4
import sqlite3
import hashlib
from typing import List
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from twilio.rest import Client
import random


app = FastAPI()

    
# Allow cross-origin resource sharing (CORS)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create SQLite database and table for users
conn = sqlite3.connect('users.db', isolation_level=None)
c = conn.cursor()
c.execute('CREATE TABLE IF NOT EXISTS users'
             '(id INTEGER PRIMARY KEY,'
             'first_name TEXT,'
             'last_name TEXT,'
             'email TEXT UNIQUE,'
             'password TEXT,'
             'specialty TEXT)')
conn.commit()


# Define user schema for signup
class UserSignUp(BaseModel):
    first_name: str
    last_name: str
    email: str
    password: str
    specialty: str

# Define user schema for login
class UserLogin(BaseModel):
    email: str
    password: str

# Signup route
@app.post("/signup")
async def signup(user: UserSignUp):
    # Check if user already exists
    c.execute("SELECT * FROM users WHERE email=?", (user.email,))
    result = c.fetchone()
    if result:
        raise HTTPException(status_code=400, detail="User already exists")

    # Generate a random 10-digit ID for the new user
    user_id = random.randint(1000000000, 9999999999)

    # Insert new user into database with the generated ID
    c.execute("INSERT INTO users (id, first_name, last_name, email, password, specialty) VALUES (?, ?, ?, ?, ?, ?)", 
              (user_id, user.first_name, user.last_name, user.email, user.password, user.specialty))
    conn.commit()
    return {"message": "User created successfully"}

# Login route
@app.post("/login")
async def login(user: UserLogin):
    # Check if user exists and password is correct
    c.execute("SELECT * FROM users WHERE email=? AND password=?", (user.email, user.password))
    result = c.fetchone()
    if result:
        return {"message": "Login successful"}
    raise HTTPException(status_code=401, detail="Invalid email or password")

# Route to retrieve user data
@app.get("/users")
async def get_users():
    c.execute("SELECT id, first_name, last_name, email, specialty FROM users")
    users = c.fetchall()
    return users

# Close the database connection when the application stops
@app.on_event("shutdown")
async def shutdown_event():
    c.close()
    conn.close()

# The remaining Twilio and SMS-related code remains the same

# Replace the following with your Twilio account SID and authentication token
account_sid = 'AC76f70e1408304d0c85e336a790b4c253'
auth_token = '86087b266e9f6a00892493e177a47dd1'
twilio_phone_number = '+13204131998'

client = Client(account_sid, auth_token)

async def send_sms(phone_number: str, message: str):
    try:
        message = client.messages.create(
            body=message,
            from_=twilio_phone_number,
            to=phone_number
        )
        return {'success': True, 'message': 'SMS sent successfully'}
    except Exception as e:
        return {'success': False, 'message': f'Error sending SMS: {str(e)}'}

@app.post('/send_sms')
async def send_sms_handler(phone_number: str, message: str):
    result = await send_sms(phone_number, message)
    return result   
 #Disable scientific notation for clarity
np.set_printoptions(suppress=True)

def crop_brain_contour(image, plot=False):
    import imutils

    # Convert the image to grayscale, and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    # Find the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    # crop new image out of the original image using the four extreme points (left, right, top, bottom)
    new_image = image[extTop[1] : extBot[1], extLeft[0] : extRight[0]]

    return new_image


def load_data(image, image_size):
    X = []
    image_width, image_height = image_size

    image = crop_brain_contour(image, plot=False)
    image = cv2.resize(image, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)
    image = image / 255.0
    X.append(image)

    X = np.array(X)

    return X


def detect_tumor_in_the_scan(img_array):
    IMG_WIDTH, IMG_HEIGHT = (240, 240)

    X = load_data(img_array, (IMG_WIDTH, IMG_HEIGHT))
    best_model = load_model(filepath="cnn-parameters-improvement-23-0.91.model")
    y = best_model.predict(X)
    Detection_Result = "The model has not detected the presence of a brain tumor in this MRI scan."
    if y[0][0] >= 0.5:
        Detection_Result = "The model has detected the presence of a brain tumor in this MRI scan."

    return Detection_Result

# Create a connection to the SQLite database
conn = sqlite3.connect('patient_history.db')
cursor = conn.cursor()

# Create the patient history table if it doesn't exist
cursor.execute('''
        CREATE TABLE IF NOT EXISTS patient_history
        (
            id TEXT PRIMARY KEY,
            national_id TEXT,
            phone_number TEXT,
            results TEXT,
            upload_times TEXT,
            age INTEGER,
            first_name TEXT,
            last_name TEXT
        )
    ''')
conn.commit()

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), age: int = None,
                     first_name: str = None,
                     last_name: str = None,
                     is_new: bool = True,
                     national_id: str = None,
                     phone_number: str = None):
    if is_new:
        if not national_id or not age or not first_name or not last_name or not phone_number:
            return {"error": "All fields (national ID, age, first name, last name, and phone number) are required for new users."}
    else:
        if not phone_number:
            return {"error": "Phone number is required."}

    if is_new:
        # Generate a new history ID based on national ID and phone number
        patient_id = hashlib.sha256((national_id + phone_number).encode()).hexdigest()
    else:
        # Generate a new history ID if national ID and phone number are not provided
        patient_id = str(uuid4())

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    classified_label = detect_tumor_in_the_scan(image)

    # Get the current date and time
    upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if is_new:
        # For new users, create a new record
        cursor.execute("INSERT INTO patient_history (id, national_id, phone_number, results, upload_times, age, first_name, last_name) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                       (patient_id, national_id, phone_number, classified_label, upload_time, age, first_name, last_name))
    else:
        # Check if a history record already exists for the patient based on national ID
        cursor.execute("SELECT * FROM patient_history WHERE national_id=?", (national_id,))
        existing_record = cursor.fetchone()

        if existing_record:
            # If a record exists, update the results and upload times
            existing_results = existing_record[3]
            existing_upload_times = existing_record[4]

            # Append the new result and upload time to the existing results and upload times
            updated_results = existing_results + "," + classified_label
            updated_upload_times = existing_upload_times + "," + upload_time

            cursor.execute("UPDATE patient_history SET results=?, upload_times=? WHERE national_id=?",
                           (updated_results, updated_upload_times, national_id))
        else:
            # If no record exists, create a new record
            cursor.execute("INSERT INTO patient_history (id, national_id, phone_number, results, upload_times, age, first_name, last_name) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                           (patient_id, national_id, phone_number, classified_label, upload_time, age, first_name, last_name))

    conn.commit()

    return {"patient_id": patient_id, "result": classified_label, "upload_time": upload_time}

@app.get("/history")
async def get_patient_history(national_id: str = None, phone_number: str = None):
    # Check if either national ID or phone number is provided
    if not national_id and not phone_number:
        return {"message": "Please provide either national ID or phone number."}

    # Query the database for the patient's history based on national ID or phone number
    if national_id:
        cursor.execute(
            "SELECT id, first_name, last_name, age, phone_number, national_id, results, upload_times FROM patient_history WHERE national_id=?",
            (national_id,)
        )
    else:
        cursor.execute(
            "SELECT id, first_name, last_name, age, phone_number, national_id, results, upload_times FROM patient_history WHERE phone_number=?",
            (phone_number,)
        )

    records = cursor.fetchall()

    if not records:
        return {"message": "No history found for the provided information."}

    # Extract the ID, first name, last name, age, phone number, national ID, result, and upload times from the records
    history = []
    for record in records:
        patient_id, first_name, last_name, age, phone_number, national_id, results, upload_times = record
        result_list = results.split(",")
        upload_times_list = upload_times.split(",")
        patient_history = {
            "patient_id": patient_id,
            "first_name": first_name,
            "last_name": last_name,
            "age": age,
            "phone_number": phone_number,
            "national_id": national_id,
            "history": result_list,
            "upload_times": upload_times_list
        }
        history.append(patient_history)

    return {"history": history}
    
