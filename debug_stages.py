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
from pyspark.ml import PipelineModel

def debug_stages():
    spark = SparkSession.builder \
        .appName("StageDebug") \
        .config("spark.driver.host", "127.0.0.1") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .config("spark.python.worker.reuse", "false") \
        .master("local[1]") \
        .getOrCreate()
    
    model = PipelineModel.load("amazon_rf_model")
    df = spark.createDataFrame([("I love this product!",)], ["Review Text"])
    
    print("Testing stages one by one...")
    current_df = df
    for i, stage in enumerate(model.stages):
        print(f"Testing Stage {i}: {type(stage).__name__}...")
        try:
            current_df = stage.transform(current_df)
            # Force a Python worker interaction if possible
            # Tokenizer/StopWordsRemover might not trigger it until collect
            res = current_df.collect()
            print(f"Stage {i} OK. Row count: {len(res)}")
        except Exception as e:
            print(f"Stage {i} FAILED: {e}")
            break
    
    spark.stop()

if __name__ == "__main__":
    debug_stages()
