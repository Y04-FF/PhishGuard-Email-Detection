# Phishing Email Detection Model

A beginner-friendly Scikit-learn project that classifies email text as **Phishing** or **Safe**.
It extracts text patterns with TF-IDF and adds simple URL-related features, then reports accuracy,
classification metrics, and a confusion matrix.

## Project contents

- `train_model.py` — loads data, trains a model, evaluates it, and saves model files.
- `app.py` — optional Streamlit web interface for checking email text.
- `data/emails.csv` — small demo dataset included so the project runs immediately.
- `requirements.txt` — Python dependencies.

## Important note about the demo data

The included CSV is a **small illustrative dataset** for demonstrating the workflow. Its accuracy
is not a reliable measure of real-world phishing detection. For a serious evaluation, replace it
with a larger, properly labelled dataset from a trusted source, remove duplicates, and use a
separate test set. Do not use this model as the only security control.

## Run locally

1. Install Python 3.10+.
2. Open a terminal in this folder.
3. Install packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Train and evaluate:

   ```bash
   python train_model.py
   ```

5. Launch the optional web app:

   ```bash
   streamlit run app.py
   ```

## Dataset format

`data/emails.csv` must have two columns:

- `text`: email subject/body or a combined email text
- `label`: `phishing` or `safe`

You can replace the demo CSV with your own licensed, labelled dataset while preserving these column names.
