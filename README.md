# Potato Disease Classification

This project aims to classify potato diseases using machine learning techniques. It provides a Python-based setup for training the model and running an API for inference. Additionally, it includes instructions for setting up a frontend ReactJs website for interacting with the model.

## Setup for Python:

1. **Install Python:** Ensure Python is installed on your system.

2. **Install Python packages:**

    ```bash
    pip3 install -r training/requirements.txt
    pip3 install -r api/requirements.txt
    ```

## Training the Model:

1. **Run Jupyter Notebook in Browser:**

    ```bash
    jupyter notebook
    ```

2. **Open `potato-disease-training.ipynb` in Jupyter Notebook.**

3. **Run all the Cells one by one.**

## Running the API:

### Using FastAPI:

1. **Navigate to the API folder:**

    ```bash
    cd api
    ```

2. **Run the FastAPI Server using uvicorn:**

    ```bash
    uvicorn main:app --reload --host localhost
    ```

    Your API is now running at `localhost:8000`.

### Using FastAPI & TF Serve:

1. **Navigate to the API folder:**

    ```bash
    cd api
    ```

2. **Run the TF Serve:**

    ```bash
    docker run -t --rm -p 8501:8501 -v /path/to/potato-disease-classification:/potato-disease-classification tensorflow/serving --rest_api_port=8501 --model_config_file=/potato-disease-classification/models.config
    ```

4. **Run the FastAPI Server using uvicorn:**

    - Run it from `main.py` or `main-tf-serving.py` using PyCharm run option.
    - Or run it from the command prompt as shown below:

    ```bash
    uvicorn main-tf-serving:app --reload --host localhost
    ```

    Your API is now running at `localhost:8000`.

## Running the Frontend:

1. **Navigate to the frontend folder:**

    ```bash
    cd frontend
    ```

2. **Install dependencies:**

    ```bash
    npm install
    ```

3. **Start the frontend ReactJs website:**

    ```bash
    npm start
    ```
