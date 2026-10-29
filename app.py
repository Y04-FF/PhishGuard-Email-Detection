from pathlib import Path
import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "phishing_model.joblib"

st.set_page_config(page_title="Phishing Email Detector", page_icon="🛡️", layout="centered")
st.title("🛡️ Phishing Email Detection")
st.write("Paste an email subject and body to get a machine-learning prediction.")
st.info("Educational demo only. Always verify suspicious messages through official channels.")

if not MODEL_PATH.exists():
    st.warning("The model has not been trained yet. Run `python train_model.py` first.")
    st.stop()

model = joblib.load(MODEL_PATH)
email_text = st.text_area(
    "Email text",
    height=220,
    placeholder="Paste the email subject and body here..."
)

if st.button("Analyze email", type="primary"):
    if not email_text.strip():
        st.error("Please enter some email text first.")
    else:
        prediction = model.predict([email_text])[0]
        probabilities = model.predict_proba([email_text])[0]
        classes = list(model.classes_)
        confidence = float(probabilities[classes.index(prediction)])
        if prediction == "phishing":
            st.error(f"⚠️ Prediction: PHISHING")
        else:
            st.success("✅ Prediction: SAFE")
        st.metric("Model confidence", f"{confidence:.1%}")
        st.caption("Confidence is the model's estimated probability, not a guarantee of safety.")
