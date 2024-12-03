from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.classification import LinearSVC, DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pandas as pd
from imblearn.over_sampling import SMOTE
from pyspark.sql import types as T
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import col

# Initialize Spark session
spark = SparkSession.builder.appName("WineQuality").getOrCreate()

# Define column names
column_names = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol", "quality"
]

# Load datasets
training_data = spark.read.csv("TrainingDataset.csv", sep=';', header=True, inferSchema=True)
validation_data = spark.read.csv("ValidationDataset.csv", sep=';', header=True, inferSchema=True)

# Rename columns
training_data = training_data.toDF(*column_names)
validation_data = validation_data.toDF(*column_names)

# Select relevant columns
selected_columns = ["citric acid", "sulphates", "alcohol", "quality"]
selected_training_data = training_data.select(*selected_columns)
selected_validation_data = validation_data.select(*selected_columns)

# Combine features into a vector
assembler = VectorAssembler(inputCols=selected_training_data.columns[:-1], outputCol="features")
assembled_training_data = assembler.transform(selected_training_data).select("features", "quality")

# Convert to Pandas for SMOTE
features_pd = assembled_training_data.select("features", "quality") \
    .rdd.map(lambda row: (row.features.toArray().tolist(), row.quality)) \
    .toDF(["features", "quality"]) \
    .toPandas()

X = pd.DataFrame(features_pd["features"].tolist(), columns=["citric acid", "sulphates", "alcohol"])
y = features_pd["quality"]

# Apply SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

# Convert back to Pandas DataFrame
resampled_df = pd.DataFrame(X_resampled, columns=["citric acid", "sulphates", "alcohol"])
resampled_df["quality"] = y_resampled.astype(float)

# Convert back to PySpark DataFrame
schema = T.StructType([
    T.StructField("citric acid", T.DoubleType(), True),
    T.StructField("sulphates", T.DoubleType(), True),
    T.StructField("alcohol", T.DoubleType(), True),
    T.StructField("quality", T.DoubleType(), True),
])
resampled_training_data = spark.createDataFrame(resampled_df, schema=schema)
resampled_training_data = assembler.transform(resampled_training_data).select("features", "quality")

# Binary Classification Conversion
threshold = 5.0  # Adjust the threshold based on your requirements
resampled_training_data = resampled_training_data.withColumn(
    "binary_quality", (resampled_training_data["quality"] >= threshold).cast("int")
)
scaled_validation_data = assembler.transform(selected_validation_data).withColumn(
    "binary_quality", (selected_validation_data["quality"] >= threshold).cast("int")
)

# Standardize the features
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withMean=True, withStd=True)
scaler_model = scaler.fit(resampled_training_data)
scaled_training_data = scaler_model.transform(resampled_training_data)
scaled_validation_data = scaler_model.transform(scaled_validation_data)

# Convert scaledFeatures (Vector) to scaledFeaturesArray (Array)
scaled_validation_data = scaled_validation_data.withColumn("scaledFeaturesArray", vector_to_array("scaledFeatures"))

# Flatten the array into individual columns
array_size = len(scaled_validation_data.select("scaledFeaturesArray").first()[0])  # Get the size of the array
feature_columns = [f"feature_{i}" for i in range(array_size)]  # Generate column names

# Add each feature as a separate column
for i, col_name in enumerate(feature_columns):
    scaled_validation_data = scaled_validation_data.withColumn(col_name, col("scaledFeaturesArray")[i])

# Select flattened features and binary_quality for writing to CSV
columns_to_save = feature_columns + ["binary_quality"]
scaled_validation_data.select(*columns_to_save).write.csv("scaled_validation_data", header=True, mode="overwrite")

# SVM
svm = LinearSVC(featuresCol="scaledFeatures", labelCol="binary_quality", maxIter=10)
svm_model = svm.fit(scaled_training_data)
svm_predictions = svm_model.transform(scaled_validation_data)

# Decision Tree
dt = DecisionTreeClassifier(featuresCol="scaledFeatures", labelCol="binary_quality", maxDepth=5)
dt_model = dt.fit(scaled_training_data)
dt_predictions = dt_model.transform(scaled_validation_data)

# Random Forest
rf = RandomForestClassifier(featuresCol="scaledFeatures", labelCol="binary_quality", numTrees=100)
rf_model = rf.fit(scaled_training_data)
rf_predictions = rf_model.transform(scaled_validation_data)

# Evaluate models
evaluator = MulticlassClassificationEvaluator(labelCol="binary_quality", metricName="f1")
print("SVM F1 Score:", evaluator.evaluate(svm_predictions))
print("Decision Tree F1 Score:", evaluator.evaluate(dt_predictions))
print("Random Forest F1 Score:", evaluator.evaluate(rf_predictions))

print("\nHighest Accuracy : ", evaluator.evaluate(rf_predictions) * 100 ,"%")

# Display predictions for Random Forest
rf_predictions.select("scaledFeatures", "binary_quality", "prediction").show(5, truncate=False)

rf_model.write().overwrite().save("wine_quality_model")