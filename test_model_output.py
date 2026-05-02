import os
import sys
import time
import shutil
import csv

# ── Environment ──
_base_dir = r'C:\Users\jprav\OneDrive\Desktop\bda_project'
os.environ['JAVA_HOME']             = os.path.join(_base_dir, 'jre17')
os.environ['HADOOP_HOME']           = os.path.join(_base_dir, 'hadoop')
os.environ['PYSPARK_PYTHON']        = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
os.environ['SPARK_LOCAL_IP']        = '127.0.0.1'

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

spark = SparkSession.builder.master("local[1]").getOrCreate()
model = PipelineModel.load("amazon_rf_model")

review_text = "Absolutely incredible! I love it so much!"
temp_in = 'test_in.csv'
temp_out = 'test_out'

if os.path.exists(temp_out): shutil.rmtree(temp_out)
with open(temp_in, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["Review Text"])
    writer.writerow([review_text])

df = spark.read.csv(temp_in, header=True)
pred = model.transform(df)
pred.select("predicted_rating", "prediction").show()
pred.select("predicted_rating").coalesce(1).write.csv(temp_out, header=True)

csv_file = [f for f in os.listdir(temp_out) if f.endswith('.csv')][0]
with open(os.path.join(temp_out, csv_file), 'r') as f:
    print("CSV Content:")
    print(f.read())

spark.stop()
