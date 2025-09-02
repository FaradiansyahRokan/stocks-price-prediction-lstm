# Stock Price Prediction with LSTM

This project implements a deep learning model (LSTM) to predict stock prices based on historical data.  
The pipeline includes **data preprocessing, normalization, model training, evaluation, and visualization**.

## Features
- Data preprocessing with MinMaxScaler
- Sequence generation (time steps windowing)
- LSTM model training with early stopping & dropout
- Model evaluation (loss, RMSE, MAE)
- Visualization of predictions vs. actual prices
- Saved scaler & model for later inference

## Tech Stack
- Python 3.x
- TensorFlow / Keras
- scikit-learn
- NumPy & Pandas
- Matplotlib

- ## How to Run

1. Clone repository
   ```bash
   git clone https://github.com/USERNAME/stock-price-prediction-lstm.git
   cd stock-price-prediction-lstm

2. Create virtual environment
   ```
   python -m venv .venv
   source .venv/bin/activate  # Mac/Linux
   .venv\Scripts\activate     # Windows

4. Install dependencies
   ```
   pip install -r requirements.txt

6. Prepare dataset
    Place your stock CSV data into data/stocks/
    Run preprocessing script:
   ```
    python scripts/01_preprocess.py
    
7. Train the model
   ```
    python scripts/02_train.py

8. Run prediction & evaluation
   ```
    python scripts/03_predict_and_evaluate.py
