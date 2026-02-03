# Analysis Workflow: Expected Points (xP) Model

This directory contains the Python scripts and generated artifacts for the NBA Expected Points (xP) and Points Over Expected (POE) analysis.

## Workflow

The analysis is divided into two main scripts that should be run in order:

1.  **`expected_points_analysis.py`**: This is the main data processing and model training script. It performs the following steps:
    *   Loads the raw 2022-23 NBA shot chart data.
    *   Generates and saves several Exploratory Data Analysis (EDA) plots (`shot_chart_...`, `shot_efficiency_heatmap_...`, `shot_distance_analysis_...`).
    *   Trains a Logistic Regression model to predict shot success probability.
    *   Saves the trained model to `expected_points_model.joblib`.
    *   Evaluates the model and prints performance metrics (Accuracy, ROC AUC, Log Loss).
    *   Calculates `xP` and `POE` for every shot in the dataset.
    *   Saves the final enriched dataset to `shots_with_xp.parquet`.

2.  **`player_performance_analysis.py`**: This is the final analysis and reporting script.
    *   Loads the enriched `shots_with_xp.parquet` dataset.
    *   Aggregates performance by player.
    *   Generates and prints leaderboards for the top and bottom 15 players by total POE.
    *   Creates and saves a POE-based shot chart for the top-performing player.

## How to Run

1.  **Setup Environment**: If you haven't already, create the Python virtual environment from the project root:
    ```bash
    python3 -m venv .venv
    ```

2.  **Install Dependencies**: Install the required packages from the project root:
    ```bash
    .venv/bin/python -m pip install -r requirements.txt
    ```
    *(If you run this from within the `analysis/` directory instead, use `../requirements.txt`.)*

3.  **Run the Data Pipeline**: Execute the main processing script.
    ```bash
    .venv/bin/python analysis/expected_points_analysis.py
    ```

4.  **Run the Final Analysis**: Execute the player-level analysis script.
    ```bash
    .venv/bin/python analysis/player_performance_analysis.py
    ```

## Generated Files

*   **`expected_points_model.joblib`**: The serialized, pre-trained Logistic Regression model.
*   **`shots_with_xp.parquet`**: The final dataset containing all shots from the 2022-23 season, enriched with xP and POE values.
*   **`*.png`**: Output visualizations from the analysis scripts.

## Database Path

By default, scripts read from `data/parsed/nba.sqlite` relative to the repo root.
You can override this via `SPATIALSPORTSR_DB_PATH`.
