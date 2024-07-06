Project Summary: SeqGAN for Synthetic Data and Anomaly Detection
Objective: Create a SeqGAN to generate synthetic patient vital signs data and integrate real-time anomaly detection to reduce false positives by 25%.

Project Structure:

Data: data/raw, data/processed, data/synthetic
Models: models/generator.py, models/discriminator.py, models/seqgan.py
Scripts: scripts/preprocess_data.py, scripts/train.py, scripts/train_anomaly_detection.py, scripts/generate_synthetic_data.py
Utilities: utils/data_utils.py, utils/model_utils.py, utils/training_utils.py
Tests: tests/test_models.py
Key Steps:

Data Preprocessing:

Normalizes data, handles missing values.
Saves processed data for training and anomaly detection.
Model Training:

Generator and Discriminator: Defines models for SeqGAN.
SeqGAN: Integrates generator and discriminator.
Training Script: Trains SeqGAN, saves the generator model.
Anomaly Detection:

Trains Isolation Forest on processed data.
Detects anomalies in generated synthetic data.
Synthetic Data Generation:

Uses trained SeqGAN generator to produce data.
Applies anomaly detection to flag data points.
Execution Commands:

Preprocess data: python scripts/preprocess_data.py
Train anomaly detection model: python scripts/train_anomaly_detection.py
Train SeqGAN model: python scripts/train.py
Generate synthetic data and detect anomalies: python scripts/generate_synthetic_data.py
