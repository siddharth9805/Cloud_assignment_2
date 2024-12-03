from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize Spark session
spark = SparkSession.builder.appName("ModelPrediction").getOrCreate()

# Load the saved Random Forest model
loaded_rf_model = RandomForestClassificationModel.load("wine_quality_model")

# Load the saved CSV
loaded_validation_data = spark.read.csv("scaled_validation_data", header=True, inferSchema=True)

# List of feature column names (update based on the number of features saved)
feature_columns = [f"feature_{i}" for i in range(len(loaded_validation_data.columns) - 1)]  # Exclude binary_quality column

# Assemble features back into a vector
assembler = VectorAssembler(inputCols=feature_columns, outputCol="scaledFeatures")
prepared_validation_data = assembler.transform(loaded_validation_data).select("scaledFeatures", "binary_quality")

# Make predictions on the prepared validation data
predictions = loaded_rf_model.transform(prepared_validation_data)

# Show predictions
predictions.select("scaledFeatures", "binary_quality", "prediction").show()

# Evaluate the model with F1 score
evaluator = MulticlassClassificationEvaluator(labelCol="binary_quality", predictionCol="prediction", metricName="f1")
f1_score = evaluator.evaluate(predictions)
print(f"Accuracy Score: {f1_score * 100}%")