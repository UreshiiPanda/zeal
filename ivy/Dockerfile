# Use an official Python runtime as the base image
FROM python:3.12-bullseye

# Set the working directory in the container
WORKDIR /

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI application code to the working directory
COPY . .

# Expose the port on which your FastAPI application runs
EXPOSE 8005

# Define the command to run your FastAPI application
CMD ["uvicorn", "ivy_live:app", "--host", "0.0.0.0", "--port", "8005", "--reload"]
#CMD ["python", "ivy_live.py"]
