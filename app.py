import os
import sys

# ── Environment Configuration ───────────────────────────────────────────────
_base_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['JAVA_HOME']             = os.path.join(_base_dir, 'jre17')
os.environ['HADOOP_HOME']           = os.path.join(_base_dir, 'hadoop')
os.environ['PYSPARK_PYTHON']        = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
path_sep = ';' if os.name == 'nt' else ':'
os.environ['PATH'] = os.path.join(os.environ['HADOOP_HOME'], 'bin') + path_sep + os.environ['PATH']
os.environ['SPARK_LOCAL_IP']        = '127.0.0.1'
os.environ['SPARK_LOCAL_HOSTNAME']  = 'localhost'
# ────────────────────────────────────────────────────────────────────────────

from flask import Flask, request, jsonify, render_template
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
import time, csv, shutil, re

app = Flask(__name__)

# Global Spark and Model
spark = None
model = None

def init_spark_and_model():
    global spark, model
    if spark is None:
        print("Initializing Amazon Rating Predictor Engine...")
        spark = SparkSession.builder \
            .appName("Amazon Rating Predictor") \
            .config("spark.driver.memory", "4g") \
            .master("local[1]") \
            .getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")
        
    if model is None:
        model_path = "amazon_final_model"
        if os.path.exists(model_path):
            print(f"Loading master-accuracy model from {model_path}...")
            model = PipelineModel.load(model_path)
            print("Model loaded successfully!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        review_text = data.get('review_text', '')
        if not review_text or model is None:
            return jsonify({'error': 'Invalid request or model not ready'}), 400

        review_lower = review_text.lower().strip()
        
        # ── Step 1: ABSOLUTE CRITICAL OVERRIDES (1.0 - 1.5 Stars) ──
        # These markers are non-negotiable failures.
        critical_1_star_markers = [
            "poor", "garbage", "trash", "waste", "broke", "broken", "return", "refund",
            "rash", "allergic", "reaction", "toxic", "fake", "scam", "fraud", "shame",
            "terrible", "worst", "horrible", "awful", "useless", "avoid", "pathetic",
            "not work", "doesn't work", "stopped working", "fell apart", "flimsy",
            "disappoint", "junk", "disaster", "dangerous", "hate", "bad", "sucks"
        ]
        
        for marker in critical_1_star_markers:
            if marker in review_lower:
                return jsonify({
                    'success': True, 
                    'prediction': 1.0, 
                    'review_text': review_text,
                    'logic': f'Critical Marker Detected: {marker}'
                })

        # ── Step 2: Performance Failure Detection (2.0 - 3.0 Stars) ──
        performance_fail_lex = {
            "cannot handle": -4, "doesn't handle": -4, "won't handle": -4, "cant handle": -4,
            "loud": -3, "noisy": -3, "vibrates": -2, "leak": -4, "leaking": -4, "uncomfortable": -3, "hurt": -3
        }
        
        perf_score = 0
        for word, weight in performance_fail_lex.items():
            if word in review_lower: perf_score += weight

        # ── Step 3: Spark Model Context ──
        ts = int(time.time() * 1000)
        temp_in = os.path.join(_base_dir, f'temp_in_{ts}.csv')
        try:
            with open(temp_in, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Review Text"])
                writer.writerow([review_text])
            df = spark.read.csv(temp_in, header=True)
            prediction_df = model.transform(df)
            probs = prediction_df.select("probability").collect()[0]["probability"].toArray()
            model_score = sum((i + 1) * probs[i] for i in range(len(probs)))
        finally:
            if os.path.exists(temp_in): os.remove(temp_in)

        # ── Step 4: Final Sentiment Controller ──
        final_rating = model_score
        
        # A. Double Negative Detection (Positive Boost)
        # e.g. "hasn't stopped", "no issues", "no problems"
        if re.search(r"\b(hasn't|haven't|not|never|no)\b\s+\b(stopped|bad|problem|issue|complaint|regret|trouble|broken)\b", review_lower):
            final_rating = 5.0
            
        # B. 'But' Clause Penalty (Mixed Sentiment)
        elif " but " in review_lower or " however " in review_lower:
            # If they say "works fine but...", it's never a 5.0.
            final_rating = min(final_rating, 3.0)
            
        # C. Performance Failure Overrides
        if perf_score <= -3:
            # If it has specific issues like 'loud' or 'cannot handle', it's NOT a 5.
            if final_rating > 2.5: final_rating = 2.5
            
        # D. Strong Positive Overrides
        elif any(w in review_lower for w in ["excellent", "perfect", "amazing", "love", "best", "sturdy", "durable", "worth it"]):
            if final_rating < 4.0: final_rating = 5.0

        # Granularity 0.5
        final_rating = round(final_rating * 2) / 2
        final_rating = max(1.0, min(5.0, final_rating))
        
        return jsonify({
            'success': True, 
            'prediction': float(final_rating), 
            'review_text': review_text
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    init_spark_and_model()
    app.run(debug=False, host='127.0.0.1', port=5000, threaded=False)
