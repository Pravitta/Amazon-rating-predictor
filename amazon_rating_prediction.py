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

from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_extract, col, length, lower
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, VectorAssembler, StringIndexer, IndexToString, NGram
from pyspark.ml import Pipeline

def main():
    print("Initializing Amazon Rating Predictor (Training Mode)...")
    spark = SparkSession.builder \
        .appName("Amazon Rating Predictor") \
        .config("spark.driver.memory", "6g") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    print("Loading full dataset...")
    file_path = "Amazon_Reviews.csv"
    df = spark.read.csv(file_path, header=True, multiLine=True, escape='"')
    
    df = df.withColumn("Label", regexp_extract(col("Rating"), r"(\d+)", 1))
    clean_df = df.filter(col("Label").isNotNull() & (col("Label") != "") & col("Review Text").isNotNull() & (length(col("Review Text")) > 5))
    
    
    target_count = 5000 
    dfs = []
    for i in range(1, 6):
        class_df = clean_df.filter(col("Label") == str(i))
        count = class_df.count()
        if count > 0:
            if count > target_count:
                class_df = class_df.sample(False, target_count / count, seed=42)
            dfs.append(class_df)
    
    balanced_df = dfs[0]
    for i in range(1, len(dfs)):
        balanced_df = balanced_df.union(dfs[i])
    
    print("Balanced Data Distribution:")
    balanced_df.groupBy("Label").count().orderBy("Label").show()
    
    
    tokenizer = Tokenizer(inputCol="Review Text", outputCol="words")
    
   
    stopWordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered_words"
    ngram2 = NGram(n=2, inputCol="filtered_words", outputCol="b_grams")
    ngram3 = NGram(n=3, inputCol="filtered_words", outputCol="t_grams")
    ngram4 = NGram(n=4, inputCol="filtered_words", outputCol="q_grams")
    
    
    hTF_1 = HashingTF(inputCol="filtered_words", outputCol="tf_1", numFeatures=10000)
    hTF_2 = HashingTF(inputCol="b_grams", outputCol="tf_2", numFeatures=10000)
    hTF_3 = HashingTF(inputCol="t_grams", outputCol="tf_3", numFeatures=10000)
    hTF_4 = HashingTF(inputCol="q_grams", outputCol="tf_4", numFeatures=10000)
    
    idf_1 = IDF(inputCol="tf_1", outputCol="idf_1")
    idf_2 = IDF(inputCol="tf_2", outputCol="idf_2")
    idf_3 = IDF(inputCol="tf_3", outputCol="idf_3")
    idf_4 = IDF(inputCol="tf_4", outputCol="idf_4")
    
    assembler = VectorAssembler(inputCols=["idf_1", "idf_2", "idf_3", "idf_4"], outputCol="features")
    
    labelIndexer = StringIndexer(inputCol="Label", outputCol="label", stringOrderType="alphabetAsc")
    

    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=200, regParam=0.005)
    
    pipeline = Pipeline(stages=[tokenizer, stopWordsRemover, ngram2, ngram3, ngram4, hTF_1, hTF_2, hTF_3, hTF_4, idf_1, idf_2, idf_3, idf_4, assembler, labelIndexer, lr])
    
    print("Training massive high-precision model...")
    model = pipeline.fit(balanced_df)
    
    print("Saving to amazon_final_model...")
    model.write().overwrite().save("amazon_final_model")
    print("Done!")
    
    spark.stop()

if __name__ == "__main__":
    main()
