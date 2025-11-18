# app.py
from flask import Flask, render_template, request
import joblib, pandas as pd, os, sqlite3
from datetime import datetime

app = Flask(__name__)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
model = joblib.load(MODEL_PATH)

DB = 'predictions.db'

def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  area_sqft REAL, bedrooms INTEGER, bathrooms INTEGER,
                  location TEXT, age INTEGER, predicted_price REAL, created_at TEXT)''')
    conn.commit()
    conn.close()

@app.route('/', methods=['GET','POST'])
def home():
    price = None
    error = None
    if request.method == 'POST':
        try:
            area = float(request.form['area_sqft'])
            bedrooms = int(request.form['bedrooms'])
            bathrooms = int(request.form['bathrooms'])
            location = request.form['location']
            age = int(request.form['age'])
        except Exception as e:
            error = "Invalid input: " + str(e)
            # fall through to render template with error

        if error is None:
            df = pd.DataFrame([{
                'area_sqft': area, 'bedrooms': bedrooms,
                'bathrooms': bathrooms, 'location': location, 'age': age
            }])
            pred = model.predict(df)[0]
            price = round(float(pred), 2)

            conn = sqlite3.connect(DB)
            c = conn.cursor()
            c.execute('''INSERT INTO predictions(area_sqft,bedrooms,bathrooms,location,age,predicted_price,created_at)
                         VALUES (?,?,?,?,?,?,?)''',
                      (area, bedrooms, bathrooms, location, age, price, datetime.now().isoformat()))
            conn.commit()
            conn.close()

    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT area_sqft,bedrooms,bathrooms,location,age,predicted_price,created_at FROM predictions ORDER BY id DESC LIMIT 5")
    recent = c.fetchall()
    conn.close()

    return render_template('index.html', price=price, recent=recent, error=error)

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
