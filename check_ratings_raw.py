import os
import sys
_base = r'C:\Users\jprav\OneDrive\Desktop\bda_project'
os.environ['JAVA_HOME'] = os.path.join(_base, 'jre17')
from pyspark.sql import SparkSession
spark = SparkSession.builder.master('local').getOrCreate()
df = spark.read.csv('Amazon_Reviews.csv', header=True, multiLine=True, escape='"')
df.select('Rating').distinct().show(truncate=False)
spark.stop()
