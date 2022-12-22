# eMIC-AntiKP: Estimating Minimum Inhibitory Concentrations of Antibiotics towards Klebsiella Pneumoniae using Deep Learning

+ Step 1. Download dataset

python3 download_data.py

+ Step 2. Calculate K-mer using kmc tool:

python3 counter.py

+ Step 3. Identify k-mers in dataset:

python3 km_counts_process.py

+ Step 4. Train and Test "Prediction MIC":

python3 prediction_MIC.py

+ Step 5. Train and Test "Resistance Prediction":

python3 resistance_prediction.py
