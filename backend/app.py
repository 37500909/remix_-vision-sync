import os
import base64
import cv2
import numpy as np
import sqlite3
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace

app = Flask(__name__)
# Enable CORS for communication from React app on port 3000
CORS(app)

DB_DIR = os.path.join(os.path.dirname(__file__), 'db')
os.makedirs(DB_DIR, exist_ok=True)

DB_FILE = os.path.join(os.path.dirname(__file__), 'biometrics.db')

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            dni TEXT,
            image_path TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    try:
        cursor.execute("ALTER TABLE users ADD COLUMN dni TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass # Column already exists
        
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS access_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            identified_name TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            emotion TEXT,
            age INTEGER,
            gender TEXT,
            success INTEGER,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    conn.commit()
    conn.close()

init_db()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'message': 'DeepFace backend is active'}), 200

@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.json
        if not data or 'image' not in data or 'first_name' not in data or 'last_name' not in data:
            return jsonify({'error': 'Image, First Name, and Last Name are required'}), 400
            
        first_name = data['first_name'].strip().replace(" ", "_")
        last_name = data['last_name'].strip().replace(" ", "_")
        
        if not first_name or not last_name:
            return jsonify({'error': 'First Name and Last Name cannot be empty'}), 400
            
        name_key = f"{first_name}_{last_name}"
        image_data = data['image']
        
        # Remove data URI prefix if present (e.g. "data:image/jpeg;base64,")
        if ',' in image_data:
            image_data = image_data.split(',')[1]
            
        img_bytes = base64.b64decode(image_data)
        
        # Save reference face image in db folder
        file_path = os.path.join(DB_DIR, f"{name_key}.jpg")
        with open(file_path, "wb") as f:
            f.write(img_bytes)
            
        dni = data.get('dni', '').strip()
        
        # Insert or update user record in local SQLite database
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE first_name = ? AND last_name = ?", (first_name, last_name))
        row = cursor.fetchone()
        if not row:
            cursor.execute(
                "INSERT INTO users (first_name, last_name, dni, image_path) VALUES (?, ?, ?, ?)",
                (first_name, last_name, dni, file_path)
            )
        else:
            cursor.execute(
                "UPDATE users SET image_path = ?, dni = ? WHERE id = ?",
                (file_path, dni, row[0])
            )
        conn.commit()
        conn.close()
            
        # Invalidate representation cache files (*.pkl) so DeepFace rebuilds its database index
        # DeepFace creates representation files inside the db_path to cache facial features.
        # Removing them forces DeepFace.find to update immediately with the new registered face.
        for file in os.listdir(DB_DIR):
            if file.endswith('.pkl'):
                try:
                    os.remove(os.path.join(DB_DIR, file))
                except Exception as e:
                    print(f"Error removing cache file {file}: {e}")
                    
        return jsonify({
            'success': True,
            'message': f'Face registered successfully as {first_name} {last_name}'
        }), 200
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
            
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
            
        # Decode base64 to numpy array for OpenCV
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Failed to decode image'}), 400
            
        # 1. Run standard DeepFace analysis for emotion, age, and gender
        try:
            results = DeepFace.analyze(
                img_path=img,
                actions=['emotion', 'age', 'gender'],
                enforce_detection=False
            )
        except Exception as ae:
            print("DeepFace analyze error:", ae)
            return jsonify({'results': [], 'identity': 'Unknown', 'message': str(ae)}), 200
            
        if not isinstance(results, list):
            results = [results]
            
        # Filter out false-positive fallback detections (whole image analyzed as face)
        h_img, w_img = img.shape[:2]
        real_results = []
        for res in results:
            region = res.get('region', {})
            # If region is exactly the size of the whole image, it means face detector failed
            # and DeepFace fell back to analyzing the entire frame.
            if region.get('x') == 0 and region.get('y') == 0 and region.get('w') == w_img and region.get('h') == h_img:
                continue
            real_results.append(res)
            
        results = real_results
        
        if len(results) == 0:
            return jsonify({'results': [], 'identity': 'Unknown', 'message': 'No face detected'}), 200
            
        # 2. Check if the face belongs to a registered user
        has_images = False
        dfs = []
        if os.path.exists(DB_DIR):
            for file in os.listdir(DB_DIR):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    has_images = True
                    break
                    
        if has_images:
            try:
                # Run DeepFace.find to search reference database
                # VGG-Face threshold is typically 0.40 for cosine distance
                dfs = DeepFace.find(
                    img_path=img,
                    db_path=DB_DIR,
                    model_name="VGG-Face",
                    enforce_detection=False
                )
                if not isinstance(dfs, list):
                    dfs = [dfs]
            except Exception as fe:
                print("DeepFace find error (ignored):", fe)
                dfs = []
                
        def find_matching_df(face_region, dfs_list):
            fx, fy, fw, fh = face_region['x'], face_region['y'], face_region['w'], face_region['h']
            for df in dfs_list:
                if df.empty:
                    continue
                row = df.iloc[0]
                sx = row.get('source_x', 0)
                sy = row.get('source_y', 0)
                # Check center distance proximity
                if abs(fx - sx) < 45 and abs(fy - sy) < 45:
                    return df
            return None

        formatted_results = []
        recognized_identities = []
        
        for res in results:
            dominant_emotion = str(res['dominant_emotion'])
            age = int(res['age'])
            dominant_gender = str(res['dominant_gender'])
            
            face_region = res['region']
            face_identity = "Unknown"
            face_dni = ""
            
            matched_df = find_matching_df(face_region, dfs)
            if matched_df is not None and not matched_df.empty:
                closest_match = matched_df.iloc[0]
                distance = closest_match.get('distance', closest_match.get('VGG-Face_cosine', 1.0))
                if distance < 0.40:
                    identity_path = closest_match['identity']
                    filename = os.path.basename(identity_path)
                    name_key = os.path.splitext(filename)[0]
                    face_identity = name_key.replace("_", " ")
                    
                    # Fetch DNI from SQLite
                    try:
                        conn = sqlite3.connect(DB_FILE)
                        cursor = conn.cursor()
                        cursor.execute("SELECT dni FROM users WHERE first_name = ? AND last_name = ?", tuple(name_key.split("_", 1)))
                        row = cursor.fetchone()
                        if row:
                            face_dni = row[0] if row[0] else ""
                        conn.close()
                    except Exception as db_err:
                        print("SQLite DNI lookup error:", db_err)
            
            if face_identity != "Unknown":
                recognized_identities.append(face_identity)
                
            formatted_results.append({
                'box': {
                    'x': int(face_region['x']),
                    'y': int(face_region['y']),
                    'w': int(face_region['w']),
                    'h': int(face_region['h'])
                },
                'dominant_emotion': dominant_emotion,
                'emotion': {k: float(v) for k, v in res['emotion'].items()},
                'age': age,
                'gender': {k: float(v) for k, v in res['gender'].items()} if isinstance(res['gender'], dict) else {},
                'dominant_gender': dominant_gender,
                'identity': face_identity,
                'dni': face_dni
            })
            
        # Log biometric attempts to database (with throttling)
        for r_res in formatted_results:
            f_identity = r_res['identity']
            f_dni = r_res['dni']
            f_emotion = r_res['dominant_emotion']
            f_age = r_res['age']
            f_gender = r_res['dominant_gender']
            success = 1 if f_identity != "Unknown" else 0
            
            user_id = None
            if success:
                parts = f_identity.split(" ")
                if len(parts) >= 2:
                    f_name, l_name = parts[0], parts[1]
                    try:
                        conn = sqlite3.connect(DB_FILE)
                        cursor = conn.cursor()
                        cursor.execute("SELECT id FROM users WHERE first_name = ? AND last_name = ?", (f_name, l_name))
                        row = cursor.fetchone()
                        if row:
                            user_id = row[0]
                        conn.close()
                    except Exception as db_err:
                        print("SQLite user lookup error:", db_err)
            
            # Throttling logic per face
            should_log = True
            try:
                conn = sqlite3.connect(DB_FILE)
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT identified_name, timestamp FROM access_logs WHERE identified_name = ? ORDER BY id DESC LIMIT 1",
                    (f_identity,)
                )
                last_row = cursor.fetchone()
                conn.close()
                
                if last_row:
                    last_name, last_time_str = last_row
                    last_time = datetime.strptime(last_time_str, "%Y-%m-%d %H:%M:%S")
                    diff = (datetime.utcnow() - last_time).total_seconds()
                    
                    if diff < 10:
                        should_log = False
            except Exception as throttle_err:
                print("SQLite throttle calculation error:", throttle_err)
                
            if should_log:
                try:
                    conn = sqlite3.connect(DB_FILE)
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO access_logs (user_id, identified_name, emotion, age, gender, success)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (user_id, f_identity, f_emotion, f_age, f_gender, success))
                    conn.commit()
                    conn.close()
                except Exception as log_err:
                    print("SQLite insert log error:", log_err)
            
        return jsonify({
            'results': formatted_results,
            'identities': recognized_identities
        }), 200
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/logs', methods=['GET'])
def get_logs():
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, identified_name, timestamp, emotion, age, gender, success 
            FROM access_logs 
            ORDER BY id DESC 
            LIMIT 20
        ''')
        rows = cursor.fetchall()
        conn.close()
        
        logs = []
        for r in rows:
            logs.append({
                'id': r[0],
                'name': r[1],
                'timestamp': r[2],
                'emotion': r[3],
                'age': r[4],
                'gender': r[5],
                'success': bool(r[6])
            })
        return jsonify({'logs': logs}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Biometric DeepFace Flask Server on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=False)
