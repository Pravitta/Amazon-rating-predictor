import os
import sys
_base = r'C:\Users\jprav\OneDrive\Desktop\bda_project'
os.environ['JAVA_HOME'] = os.path.join(_base, 'jre17')
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
spark = SparkSession.builder.master('local').getOrCreate()
model = PipelineModel.load('amazon_rf_model')
print("Model Stages:")
for i, stage in enumerate(model.stages):
    print(f"Stage {i}: {type(stage).__name__}")
    if "LogisticRegression" in type(stage).__name__:
        print(f"  Coefficients: {stage.coefficientMatrix}")
    if "IndexToString" in type(stage).__name__:
        print(f"  Labels: {stage.getLabels()}")
spark.stop()
