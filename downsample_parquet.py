from pyspark.sql import SparkSession

# converts parquet files to spark dataframes, downsample, then save to machine
def save_downsample_parquet(spark, frac):
    data_train = spark.read.parquet('hdfs:/user/bm106/pub/MSD/cf_train.parquet')
    data_val = spark.read.parquet('hdfs:/user/bm106/pub/MSD/cf_validation.parquet')
    data_test = spark.read.parquet('hdfs:/user/bm106/pub/MSD/cf_test.parquet')
    
    df_train = data_train.select('user_id','track_id','count')
    df_val = data_val.select('user_id','track_id','count')
    df_test = data_test.select('user_id','track_id','count')

    df_train = data_train.sample(withReplacement=False, fraction=frac)
    df_val = data_val.sample(withReplacement=False, fraction=frac)
    df_test = data_test.sample(withReplacement=False, fraction=frac)

    df_train.write.parquet('hdfs:/user/jxl219/cf_train.parquet')
    df_val.write.parquet('hdfs:/user/jxl219/cf_validation.parquet')
    df_test.write.parquet('hdfs:/user/jxl219/cf_test.parquet')

def main(spark):
    save_downsample_parquet(spark, 0.01)
 
    
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('downsample_parquet.py').getOrCreate()

    main(spark)
