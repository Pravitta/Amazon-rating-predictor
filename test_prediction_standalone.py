import os
import sys

# ── Environment must be set BEFORE pyspark is imported ──────────────────────
_base_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['JAVA_HOME']             = os.path.join(_base_dir, 'jre17')
os.environ['HADOOP_HOME']           = os.path.join(_base_dir, 'hadoop')
os.environ['PYSPARK_PYTHON']        = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Add hadoop/bin to PATH for winutils.exe
path_sep = ';' if os.name == 'nt' else ':'
os.environ['PATH'] = os.path.join(os.environ['HADOOP_HOME'], 'bin') + path_sep + os.environ['PATH']

os.environ.setdefault('SPARK_LOCAL_IP',       '127.0.0.1')
os.environ.setdefault('SPARK_LOCAL_HOSTNAME', 'localhost')
# ────────────────────────────────────────────────────────────────────────────

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.types import StructType, StructField, StringType

def test_prediction():
    print("Initializing Spark Session for test...")
    spark = SparkSession.builder \
        .appName("PredictionTest") \
        .config("spark.driver.host", "127.0.0.1") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .config("spark.python.worker.faulthandler.enabled", "true") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
        .master("local[1]") \
        .getOrCreate()
    
    model_path = "amazon_rf_model"
    print(f"Loading model from {model_path}...")
    model = PipelineModel.load(model_path)
    
    review_text = "This is a test review. I love this product!"
    print(f"Testing prediction for: {review_text}")
    
    schema = StructType([StructField("Review Text", StringType(), True)])
    df = spark.createDataFrame([(review_text,)], schema)
    
    prediction_df = model.transform(df)
    print("Transform complete, calling first()...")
    
    try:
        predicted_rating = prediction_df.select("prediction").first()[0]
        print(f"SUCCESS! Predicted rating: {predicted_rating}")
    except Exception as e:
        print(f"FAILED! Error: {e}")
    
    spark.stop()

if __name__ == "__main__":
    test_prediction()
