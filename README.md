# Overview
This project implements an Intrusion Detection System (IDS) using:
- WiSARD weightless neural network  
- Random Forest  
- Linear SVM  
- KNN  
Dataset used: IoT-SDN IDS Dataset (local CSV copy due to Kaggle authentication).

## 1. Prerequisites
Ensure the following are installed:
- Python 3.9+
- pip  
- git

## 2. Create and Activate Virtual Environment (Windows)
Before activating a virtual environment in PowerShell, enable script execution **once**:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy Unrestricted -Force
```

Now create the environment:
```powershell
python -m venv venv
```

Activate:
```powershell
venv\Scripts\activate
```

When activated successfully, the terminal will start with (venv)

## 3. Install Dependencies
With the venv active:
```powershell
pip install -r requirements.txt
```

## 4. Project Structure
```
ids-wisard/
│
├── experiment.py          # Main orchestration script
├── wisard_model.py        # WiSARD implementation
├── datasetutils.py        # Dataset loading + cleaning
├── binarization.py        # Normalization + thermometer encoding
├── baselines.py           # Random Forest, SVM and KNN
│
├── data/
│   └── dataset.csv        # IoT-SDN IDS dataset
│
└── README.md
```

## 5. Running the Experiment
Inside the project folder, with the virtual environment active, run the following command:
```powershell
python experiment.py
```

The script will automatically:
- Load and clean the dataset
- Clip outliers
- Remove low-variance features
- Normalize the data
- Convert to binary vectors (thermometer encoding)
- Apply bit scrambling
- Perform hyperparameter grid search (N_BITS, tuple_size)
- Train WiSARD
- Train baseline models
- Print accuracy comparison

## 6. Notes
WiSARD accuracy is strongly affected by preprocessing.
Improvements applied in this project:
- Outlier clipping
- Hyperparameter tuning
- Low-variance feature removal
- Scrambling of binary vectors

## 7. Customizing the Grid Search
Inside the file `experiment.py`, modify (for integers):
```python
bits_options = [value1, value2, value3, value4, value5] 
tuple_sizes = [value1, value2, value3, value4, value5]
```

## 8. Questions or Help
If you need:
- graphs
- accuracy tables
- report methodology sections
Just ask me.
