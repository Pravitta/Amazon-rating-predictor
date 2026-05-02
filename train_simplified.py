import os
import sys

# ── Environment must be set BEFORE pyspark is imported ──────────────────────
_base_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['JAVA_HOME']             = os.path.join(_base_dir, 'jre17')
os.environ['HADOOP_HOME']           = os.path.join(_base_dir, 'hadoop')
os.environ['PYSPARK_PYTHON']        = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
path_sep = ';' if os.name == 'nt' else ':'
os.environ['PATH'] = os.path.join(os.environ['HADOOP_HOME'], 'bin') + path_sep + os.environ['PATH']
os.environ['SPARK_LOCAL_IP']        = '127.0.0.1'
# ────────────────────────────────────────────────────────────────────────────

from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline
from pyspark.sql.functions import regexp_extract, col, length

def train_simplified():
    print("Initializing Spark for simplified training...")
    spark = SparkSession.builder \
        .appName("SimplifiedTrain") \
        .config("spark.driver.host", "127.0.0.1") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .master("local[1]") \
        .getOrCreate()
    
    print("Loading data...")
    df = spark.read.csv("Amazon_Reviews.csv", header=True, inferSchema=True, multiLine=True, escape='"')
    df = df.withColumn("Label", regexp_extract(col("Rating"), r"(\d+)", 1).cast("double"))
    clean_df = df.filter(col("Label").isNotNull() & col("Review Text").isNotNull() & (length(col("Review Text")) > 0))
    
    print("Building simplified pipeline (NO StopWordsRemover)...")
    tokenizer = Tokenizer(inputCol="Review Text", outputCol="words")
    # Link words DIRECTLY to hashingTF
    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=1000)
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    assembler = VectorAssembler(inputCols=["features"], outputCol="final_features")
    rf = RandomForestRegressor(featuresCol="final_features", labelCol="Label", numTrees=5, maxDepth=5)
    
    pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, assembler, rf])
    
    print("Training model...")
    model = pipeline.fit(clean_df)
    
    model_path = "amazon_rf_model_simple"
    print(f"Saving model to {model_path}...")
    model.write().overwrite().save(model_path)
    
    print("Testing single prediction...")
    test_df = spark.createDataFrame([("Great product!",)], ["Review Text"])
    pred = model.transform(test_df).collect()[0]["prediction"]
    print(f"Prediction OK: {pred}")
    
    spark.stop()

if __name__ == "__main__":
    train_simplified()
