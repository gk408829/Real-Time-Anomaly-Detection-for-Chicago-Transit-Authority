# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /code

# Copy the requirements file into the container at /code
COPY ./requirements.txt /code/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the local 'app' directory into the container at /code/app
COPY ./app /code/app

# Copy the local 'mlruns' directory (containing the model) into the container
COPY ./mlruns /code/mlruns

# Command to run the application when the container launches
# Uvicorn will run the FastAPI app located in /code/app/main.py
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
