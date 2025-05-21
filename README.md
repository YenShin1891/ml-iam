# IAM Emulation with ML

This repository contains a machine learning pipeline for Integrated Assessment Model (IAM) emulation. 

## Installation
### Prerequisites
- Python 3.9
- CUDA 11.6
- Streamlit port 8501-8509

### Install Dependencies
Using `pip`:
```bash
pip install -r requirements.txt
```

## Usage
**Configuration**

Update paths and constants in configs/config.py to match your environment.

**Training and Testing on Background**

Run the training and testing pipeline:
```
nohup python scripts/train_xgb.py &
```
**Running the Dashboard**

To start the Streamlit dashboard in the background with logging enabled, run:
```
nohup streamlit run scripts/dashboard.py --logger.level=info --server.runOnSave=false &
```

## Project Structure
```
├── configs/
│   ├── config.py          # Configuration file
├── scripts/
│   ├── train_xgb.py     # Training and testing script
│   ├── dashboard.py       # Streamlit dashboard
├── src/
│   ├── data/              # Data preprocessing
│   ├── trainers/          # Model training and evaluation
│   ├── utils/             # Utility functions and plotting
├── requirements.txt     # Dependencies for pip
├── README.md              # Project documentation
```
