# ---------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------
import os
import uuid
import numpy as np
import sqlite3
import cv2
import tensorflow as tf
from datetime import datetime

from flask import Flask, render_template, request, redirect, flash, session
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from utils.report_generator import generate_report

# ---------------------------------------------------------------
# FLASK CONFIG
# ---------------------------------------------------------------
app = Flask(__name__)
app.secret_key = "supersecretkey"

UPLOAD_FOLDER = "static"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ---------------------------------------------------------------
# DATABASE
# ---------------------------------------------------------------
def get_db_connection():
    conn = sqlite3.connect("database.db")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()

    # USERS
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT UNIQUE,
            password TEXT
        )
    """)

    # PREDICTIONS
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            result TEXT,
            confidence REAL,
            image_path TEXT,
            created_at TEXT
        )
    """)

    conn.commit()
    conn.close()


init_db()

# ---------------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------------
MODEL_PATH = "models/best_model.h5"
model = load_model(MODEL_PATH)

def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            return layer.name
    return None

last_conv_name = find_last_conv_layer(model)

# ---------------------------------------------------------------
# GRAD-CAM
# ---------------------------------------------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8

    return heatmap

# ---------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")

# ---------------- REGISTER ----------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        password = generate_password_hash(request.form["password"])

        try:
            conn = get_db_connection()
            conn.execute(
                "INSERT INTO users(name,email,password) VALUES (?,?,?)",
                (name, email, password)
            )
            conn.commit()
            conn.close()
            flash("Registration successful", "success")
            return redirect("/login")
        except:
            flash("Email already exists", "danger")

    return render_template("register.html")

# ---------------- LOGIN ----------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        conn = get_db_connection()
        user = conn.execute(
            "SELECT * FROM users WHERE email=?", (email,)
        ).fetchone()
        conn.close()

        if user and check_password_hash(user["password"], password):
            session["user_id"] = user["id"]
            session["user_name"] = user["name"]
            return redirect("/dashboard")

        flash("Invalid credentials", "danger")

    return render_template("login.html")

# ---------------- LOGOUT ----------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

# ---------------- DASHBOARD ----------------
@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect("/login")

    conn = get_db_connection()

    total_predictions = conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE user_id=?",
        (session["user_id"],)
    ).fetchone()[0]

    last_prediction = conn.execute(
        "SELECT result FROM predictions WHERE user_id=? ORDER BY id DESC LIMIT 1",
        (session["user_id"],)
    ).fetchone()

    history = conn.execute("""
        SELECT result, confidence, image_path, created_at
        FROM predictions
        WHERE user_id=?
        ORDER BY id DESC
        LIMIT 5
    """, (session["user_id"],)).fetchall()

    conn.close()

    return render_template(
        "dashboard.html",
        name=session["user_name"],
        total_predictions=total_predictions,
        last_result=last_prediction["result"] if last_prediction else "—",
        history=history
    )


# ---------------- PREDICT ----------------
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "user_id" not in session:
        return redirect("/login")

    if request.method == "POST":
        file = request.files["image"]
        filename = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
        file.save(filepath)

        # PREPROCESS IMAGE
        img = image.load_img(filepath, target_size=(224, 224))
        img_arr = image.img_to_array(img) / 255.0
        img_arr = np.expand_dims(img_arr, axis=0)

        # MODEL PREDICTION
        prediction = model.predict(img_arr)[0][0]
        confidence = round(float(prediction) * 100, 2)
        result = "Malignant" if prediction >= 0.5 else "Benign"

        # SAVE TO DATABASE
        conn = get_db_connection()
        conn.execute("""
            INSERT INTO predictions(user_id, result, confidence, image_path, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            session["user_id"],
            result,
            confidence,
            unique_name,
            datetime.now().strftime("%Y-%m-%d %H:%M")
        ))
        conn.commit()
        conn.close()

        # -------- GRAD-CAM --------
        heatmap = make_gradcam_heatmap(img_arr, model, last_conv_name)
        heatmap = cv2.resize(heatmap, (224, 224))
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        superimposed = heatmap * 0.4 + img_arr[0] * 255
        gradcam_name = f"gradcam_{uuid.uuid4().hex}.png"
        cv2.imwrite(
            os.path.join(app.config["UPLOAD_FOLDER"], gradcam_name),
            np.uint8(superimposed)
        )

        # -------- REPORT GENERATION --------
        report_file = generate_report(
            session["user_name"],
            result,
            confidence,
            unique_name
        )

        return render_template(
            "predict.html",
            image_file=unique_name,
            gradcam_file=gradcam_name,
            result=result,
            confidence=confidence,
            report_file=report_file
        )

    return render_template("predict.html")

# ---------------- FORGOT PASSWORD ----------------
@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        flash("Password reset link feature will be added soon.", "info")
    return render_template("forgot_password.html")

# ---------------------------------------------------------------
# ---------------- PROFILE ----------------
@app.route("/profile")
def profile():
    if "user_id" not in session:
        return redirect("/login")

    conn = get_db_connection()
    user = conn.execute(
        "SELECT name, email FROM users WHERE id=?",
        (session["user_id"],)
    ).fetchone()
    conn.close()

    return render_template("profile.html", user=user)
#---------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
