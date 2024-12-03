
# Wine Quality Prediction With Spark

---

## Table of Contents
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Setup and Requirements](#setup-and-requirements)
   - [Apache Spark Setup on AWS EC2](#apache-spark-setup-on-aws-ec2)
   - [Docker Setup on EC2](#docker-setup-on-ec2)
4. [Model Training and Prediction](#model-training-and-prediction)
5. [Usage Instructions](#usage-instructions)
6. [Contributors](#contributors)
7. [License](#license)
---

## Introduction
This project implements a wine quality prediction machine learning (ML) model using Apache Spark on AWS. The model is trained in parallel on four EC2 instances using Spark's MLlib and deployed for predictions in a Docker container on a single EC2 instance. The model achieves 94% accuracy with a Random Forest classifier.

---

## Project Structure
```
wine-quality-prediction/
│
├── train.py              # Code for model training
├── predict.py            # Code for wine quality prediction
├── Dockerfile            # Docker configuration for the prediction application
├── datasets/
│   ├── TrainingDataset.csv  # Training data
│   ├── ValidationDataset.csv # Validation data
│
└── README.md            # Project documentation
```

---

## Setup and Requirements

### Apache Spark Setup on AWS EC2

Follow these steps to set up Apache Spark on an AWS EC2 instance:

1. **Update System and Install Java**  
   Update the system and install OpenJDK 1.8, as Spark requires Java to run.
   ```bash
   sudo yum update -y
   sudo yum install java-1.8.0-openjdk -y
   ```

2. **Download and Install Spark**  
   Download the Apache Spark binary distribution and extract it.
   ```bash
   wget https://dlcdn.apache.org/spark/spark-3.5.3/spark-3.5.3-bin-hadoop3.tgz
   tar -xzf spark-3.5.3-bin-hadoop3.tgz
   sudo mv spark-3.5.3-bin-hadoop3 /usr/local/spark
   ```

3. **Set Environment Variables**  
   Configure the environment variables for Spark.
   ```bash
   echo "export SPARK_HOME=/usr/local/spark" >> ~/.bashrc
   echo "export PATH=\$SPARK_HOME/bin:\$PATH" >> ~/.bashrc
   source ~/.bashrc
   ```

4. **Configure Spark Master**  
   Set up the Spark Master by adding its configuration.
   ```bash
   echo "SPARK_MASTER_HOST='172.31.88.119'" >> /usr/local/spark/conf/spark-env.sh
   chmod +x /usr/local/spark/conf/spark-env.sh
   /usr/local/spark/sbin/start-master.sh
   ```

5. **Configure Spark Worker**  
   Set up the Spark Worker with desired resources.
   ```bash
   echo "SPARK_WORKER_CORES=1" >> /usr/local/spark/conf/spark-env.sh
   echo "SPARK_WORKER_MEMORY=2g" >> /usr/local/spark/conf/spark-env.sh
   chmod +x /usr/local/spark/conf/spark-env.sh
   /usr/local/spark/sbin/start-worker.sh spark://172.31.88.119:7077
   ```

6. **Install Python and PySpark**  
   Install Python 3 and PySpark.
   ```bash
   sudo yum install python3 -y
   sudo pip3 install pyspark
   ```

7. **Verify PySpark Connection**  
   Test the connection to the Spark cluster.
   ```bash
   pyspark --master spark://172.31.88.119:7077
   ```

8. **Submit Spark Jobs**  
   Submit jobs to the Spark cluster.
   ```bash
   spark-submit --master spark://172.31.88.119:7077 train.py
   spark-submit --executor-cores 1 --executor-memory 1g --total-executor-cores 2 train.py
   ```

9. **Restart Spark Services**  
   Restart Spark Master and Worker if needed.
   ```bash
   /usr/local/spark/sbin/stop-all.sh
   /usr/local/spark/sbin/start-all.sh
   ```

10. **Edit Spark Configuration**  
    Edit the Spark configuration file as needed.
    ```bash
    nano $SPARK_HOME/conf/spark-env.sh
    ```

11. **Access Spark Web UI**  
    Access the Spark Web UI for monitoring and management:
    ```
    http://<master_public_ip>:8080
    ```

### Docker Setup on EC2

Follow these steps to set up Docker on an Amazon Linux 2 EC2 instance:

1. **Connect to Your EC2 Instance**
   ```bash
   ssh -i "your-key.pem" ec2-user@your-ec2-instance-public-ip
   ```

2. **Update the System**
   ```bash
   sudo yum update -y
   ```

3. **Install Docker**
   ```bash
   sudo amazon-linux-extras enable docker
   sudo yum install docker -y
   ```

4. **Start Docker Service**
   ```bash
   sudo systemctl start docker
   sudo systemctl enable docker
   ```

5. **Add User to the Docker Group**
   ```bash
   sudo usermod -aG docker ec2-user
   exit
   ssh -i "your-key.pem" ec2-user@your-ec2-instance-public-ip
   ```

6. **Verify Docker Installation**
   ```bash
   docker --version
   docker run hello-world
   ```

7. **Install Docker Compose**
   ```bash
   sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
   sudo chmod +x /usr/local/bin/docker-compose
   docker-compose --version
   ```

8. **Tagging and Pushing Docker Images**
   ```bash
   docker tag siddharth_wine_predictions:latest sidhero/siddharth_wine_predictions:latest
   docker push sidhero/siddharth_wine_predictions:latest
   ```

---

## Model Training and Prediction

1. **Parallel Training**:
   - Use `train.py` to train the model in parallel across 4 EC2 instances.

2. **Prediction Application**:
   - Use `predict.py` to predict wine quality.
   - The application runs on EC2, either directly or within a Docker container.

---

## Usage Instructions

### Training the Model
1. Upload `TrainingDataset.csv` to the Spark master node.
2. Run the training script:
   ```bash
   spark-submit --master spark://172.31.88.119:7077 train.py
   spark-submit --executor-cores 1 --executor-memory 1g --total-executor-cores 2 train.py
   ```

### Running the Prediction Application
1. **Without Docker**:
   ```bash
   python3 predict.py --input ValidationDataset.csv
   ```

2. **With Docker**:
   ```bash
   docker pull sidhero/siddharth_wine_predictions
   docker run sidhero/siddharth_wine_predictions
   ```
---

## Contributors
- Siddharth Umachandar - Developer and Maintainer

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.