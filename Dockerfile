# Use Python as the base image
FROM python:3.8-slim

# Copy application files
COPY . /app/
WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y default-jdk && \
    apt-get clean

# Set JAVA_HOME environment variable
ENV JAVA_HOME="/usr/lib/jvm/java-17-openjdk-amd64/"
ENV PATH="$JAVA_HOME/bin:$PATH"

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (if required)
EXPOSE 5000

# Command to run the application
CMD ["python", "predict.py"]