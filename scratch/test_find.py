import os
from deepface import DeepFace

DB_DIR = r"c:\Users\andres.salgado\Downloads\remix_-vision-sync\backend\db"
img_path = os.path.join(DB_DIR, "andres_salgado.jpg")

if os.path.exists(img_path):
    print("Found image:", img_path)
    try:
        dfs = DeepFace.find(
            img_path=img_path,
            db_path=DB_DIR,
            model_name="VGG-Face",
            enforce_detection=False
        )
        print("Result type:", type(dfs))
        if len(dfs) > 0:
            df = dfs[0]
            print("DataFrame columns:", df.columns.tolist())
            if not df.empty:
                print("First row:\n", df.iloc[0])
    except Exception as e:
        print("Error during test:", e)
else:
    print("Image not found at:", img_path)
