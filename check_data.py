import os
import sys
_base = r'C:\Users\jprav\OneDrive\Desktop\bda_project'
os.environ['JAVA_HOME'] = os.path.join(_base, 'jre17')
from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_extract, col
spark = SparkSession.builder.master('local').getOrCreate()
df = spark.read.csv('Amazon_Reviews.csv', header=True, multiLine=True, escape='"')
df = df.withColumn('Label', regexp_extract(col('Rating'), r'(\d+)', 1).cast('double'))
df.groupBy('Label').count().orderBy('Label').show()
spark.stop()
