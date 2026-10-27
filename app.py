import streamlit as st
import statsmodels.api as sm
import numpy as np

# Load the trained statsmodels OLS model and cache it for performance
@st.cache_resource
def load_model():
    return sm.load("model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load model.pkl. Ensure the file is in the same directory. Error: {e}")
    st.stop()

st.title("OLS Regression Predictor")
st.markdown("Enter your independent variables to generate a prediction.")

# Input field for features
features_input = st.text_input(
    "Feature Values (comma-separated)", 
    placeholder="e.g., 1.2, 3.4, 5.6"
)

if st.button("Predict"):
    if features_input:
        try:
            # Parse the comma-separated string into a list of floats
            features = [float(x.strip()) for x in features_input.split(",")]
            features_array = np.array(features)
            
            # If your OLS model was trained with an explicit intercept via sm.add_constant:
            # features_array = sm.add_constant(features_array, has_constant='add')
            
            # Generate the prediction
            prediction = model.predict(features_array)
            
            # Display result
            st.success(f"**Predicted Value:** {prediction[0]:.4f}")
            
        except ValueError:
            st.error("Please enter valid numerical values separated by commas.")
        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        st.warning("Please enter feature values before predicting.")