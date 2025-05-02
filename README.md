# Natural Disaster Prediction Dashboard

This project uses an attention-based LSTM model to predict natural disasters and provides an interactive dashboard for visualization.

## Project Structure
```
├── models/              # Saved model files
├── data/               # Dataset
├── src/                # Source code
│   ├── train.py       # Model training script
│   └── app.py         # Streamlit dashboard
├── requirements.txt    # Dependencies
└── README.md          # Project documentation
```

## Setup and Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python src/train.py
```

3. Run the dashboard:
```bash
streamlit run src/app.py
```

## Features

- Disaster prediction for different countries
- Interactive visualizations
- Model performance metrics
- Historical data analysis
