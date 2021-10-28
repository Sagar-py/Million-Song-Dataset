from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import pyspark.sql.functions as F

from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import col, explode
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.types import *
import itertools

# converts parquet files to spark dataframes
def process_parquet(spark, frac):
    data_train = spark.read.parquet('hdfs:/user/bm106/pub/MSD/cf_train.parquet')
    data_val = spark.read.parquet('hdfs:/user/bm106/pub/MSD/cf_validation.parquet')
    data_test = spark.read.parquet('hdfs:/user/bm106/pub/MSD/cf_test.parquet')
    
    df_train = data_train.select('user_id','track_id','count')

    df_val = data_val.select('user_id','track_id','count')

    df_test = data_test.select('user_id','track_id','count')

    # subsample % of data first
    if frac != 1:
        df_train = data_train.sample(withReplacement=False, fraction=frac)
        df_val = data_val.sample(withReplacement=False, fraction=frac)
        df_test = data_test.sample(withReplacement=False, fraction=frac)

    return df_train, df_val, df_test


def evaluate(spark, predictions, als_model, user_data):
  
    predictions = predictions.select('user_id','track_id','count')
    predictions.createOrReplaceTempView("predictions")
    
    ground_truth = spark.sql('SELECT user_id, collect_list(track_id) AS truth_tracks FROM predictions GROUP BY user_id')
    ground_truth.createOrReplaceTempView('ground_truth')
    
    #  get all distinct users from val dataset
    val_users = user_data.select('user_id').distinct()
    
    # get 500 recommendations each for users in the val dataset 
    recommended = als_model.recommendForUserSubset(val_users, 500)
    recommended.createOrReplaceTempView("recommended")
  
    explode_recommended = (recommended.select("user_id", explode("recommendations").alias("recommendation")).select("user_id", "recommendation.*"))
    explode_recommended.createOrReplaceTempView("explode_recommended")
    
    # collect recommended tracks for each user from recommendations
    agg_recommended = spark.sql('SELECT user_id, collect_list(track_id) AS recommended_tracks FROM explode_recommended GROUP BY user_id')
    agg_recommended.createOrReplaceTempView("agg_recommended")

    # join ground truth and recommended and choose only tracks recommended/ground truth for each user_id 
    ground_truth_recommended = spark.sql('SELECT agg_recommended.recommended_tracks AS recommended_tracks, ground_truth.truth_tracks as truth_tracks FROM agg_recommended INNER JOIN ground_truth ON agg_recommended.user_id = ground_truth.user_id')
    
    # get RDD to pass to RankingMetrics 
    ground_truth_recommended_rdd = ground_truth_recommended.select("recommended_tracks", "truth_tracks").rdd
    
    ranking_metrics = RankingMetrics(ground_truth_recommended_rdd)
    
    # Compute Ranking Metrics
    
    precision_at_K = ranking_metrics.precisionAt(500)
    mean_average_precision = ranking_metrics.meanAveragePrecision
    ncdg_at_K = ranking_metrics.ndcgAt(500)
    
    return precision_at_K, mean_average_precision, ncdg_at_K


def main(spark):
    
    df_train, df_val, df_test = process_parquet(spark, 0.05)


    # Since ALS only needs data to be in int but user_id and track_id are in string. So, we hash user_id and track_id.

    df_train = df_train.withColumn('user_id', F.hash(col('user_id')))
    df_train = df_train.withColumn('track_id', F.hash(col('track_id')))
    df_val = df_val.withColumn('user_id', F.hash(col('user_id')))
    df_val = df_val.withColumn('track_id', F.hash(col('track_id')))
    df_test = df_test.withColumn('user_id', F.hash(col('user_id')))
    df_test = df_test.withColumn('track_id', F.hash(col('track_id')))

    reg_param_list = [0.01, 0.05, 0.1]
    rank_param_list = [5, 10 , 20, 25, 50]
    alpha_param_list = [1, 10, 15]

    param_combinations = itertools.product(rank_param_list, reg_param_list, alpha_param_list)

    best_rank = None
    best_reg_param = None
    best_alpha = None
    best_mean_average_precision = float("-inf")

    for rank, reg_param, alpha in param_combinations:

        als = ALS(seed = 1, rank = rank, regParam = reg_param, alpha = alpha, userCol = "user_id", itemCol = "track_id", ratingCol = "count", implicitPrefs = True, coldStartStrategy="drop")

        #  Fit the model
        als_model = als.fit(df_train)
        
        # get predictions
        predictions = als_model.transform(df_val)    
        
        precision_at_K, mean_average_precision, ncdg_at_K = evaluate(spark, predictions, als_model, df_val)
        
        print("trained model with:")
        print("reg_param = ", reg_param)
        print("rank = ",rank)
        print("alpha =", alpha)
        print("Precision At K = ", precision_at_K)
        print("Mean Average Precision = ", mean_average_precision)
        print("ncdg At K = ", ncdg_at_K)

        if mean_average_precision > best_mean_average_precision:
        
            best_mean_average_precision = mean_average_precision
            
            best_rank = rank
            best_reg_param = reg_param
            best_alpha = alpha

    print("Best model:")
    print("best_reg_param", best_reg_param)
    print("best_rank", best_rank)
    print("best_alpha", best_alpha)
    print("best_mean_average_precision", best_mean_average_precision)




if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('ALS').getOrCreate()

    main(spark)
