# Student Social Media Addiction Analysis

## Project Overview
This project analyzes the impact of social media usage on students' lives, focusing on academic performance, mental health, and addiction levels. Using the `Students Social Media Addiction.csv` dataset, we perform Exploratory Data Analysis (EDA) to visualizations trends and unsupervised machine learning (Random Forest Regressor) to predict addiction scores.

## Dataset
The dataset contains information on 700+ students, including:
- **Demographics:** Age, Gender, Country.
- **Usage:** Average Daily Usage Hours, Most Used Platform.
- **Impact:** Effects on Academic Performance, Sleep Hours, Mental Health Score.
- **Target Variable:** `Addicted_Score` (0-10 scale).

## Files in the Repository
- **`Students Social Media Addiction.csv`**: The raw dataset used for analysis.
- **`Social_Media_Addiction_Analysis.ipynb`**: The main Jupyter Notebook containing the full code for data cleaning, EDA, and model building.
- **`Social_Media_Addiction_Analysis.html`**: An HTML export of the notebook for easy viewing in a browser.
- **`run_analysis.py`**: A Python script to quickly run the model and see performance metrics in the terminal.
- **`model_predictions.csv`**: A CSV file generated containing the test dataset with both Actual and Predicted Addiction Scores.

## Prerequisites
To run this project, you need Python installed with the following libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn notebook
```

## How to Run

### Option 1: Jupyter Notebook (Recommended)
This allows you to see the code, analysis, and interactive plots.
1. Open your terminal or command prompt.
2. Navigate to the project directory.
3. Run the following command:
   ```bash
   jupyter notebook
   ```
4. Click on `Social_Media_Addiction_Analysis.ipynb` to open it.

### Option 2: Python Script
To quickly validate the model and see the output metrics:
```bash
python run_analysis.py
```

## Key Results
The Random Forest Regressor model achieved the following performance on the test set:
- **Mean Squared Error (MSE):** ~0.044
- **R² Score:** ~0.98

These scores indicate that the model is highly accurate in predicting the addiction score based on the provided student data.

## Visualizations
The analysis includes several plots (available in the Notebook/HTML):
- Distribution of Addiction Scores.
- Correlation Heatmap.
- Feature Importance (identifying which factors contribute most to addiction).
- Usage Patterns by Gender and Platform.
