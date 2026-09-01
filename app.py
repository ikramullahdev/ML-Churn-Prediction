import streamlit as st
import joblib
import pandas as pd


st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)


st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666;
        margin-bottom: 25px;
    }

    .risk-high {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 28px;
        font-weight: 700;
        border: 2px solid #ff4b4b;
    }

    .risk-medium {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 28px;
        font-weight: 700;
        border: 2px solid #ffa500;
    }

    .risk-low {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 28px;
        font-weight: 700;
        border: 2px solid #21c354;
    }

    .info-box {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #ddd;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def load_model():

    return joblib.load(
        "models/churn_logistic_pipeline.pkl"
    )


try:

    model = load_model()

except Exception as e:

    st.error(" Model could not be loaded.")

    st.code(str(e))

    st.stop()


st.markdown(
    '<div class="main-title">📊 Customer Churn Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning powered customer churn risk prediction system'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


st.subheader(" Model Information")

info1, info2, info3, info4 = st.columns(4)

with info1:
    st.metric(
        "Model",
        "Logistic Regression"
    )

with info2:
    st.metric(
        "ROC-AUC",
        "0.842"
    )

with info3:
    st.metric(
        "F1 Score",
        "0.607"
    )

with info4:
    st.metric(
        "Decision Threshold",
        "0.25"
    )


st.divider()


st.subheader("👤 Customer Information")

col1, col2, col3 = st.columns(3)


with col1:

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.number_input(
        "Tenure (Months)",
        min_value=0,
        max_value=72,
        value=12,
        step=1
    )


with col2:

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        [
            "No",
            "Yes",
            "No phone service"
        ]
    )

    internet_service = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )

    online_security = st.selectbox(
        "Online Security",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    online_backup = st.selectbox(
        "Online Backup",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )


with col3:

    device_protection = st.selectbox(
        "Device Protection",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    tech_support = st.selectbox(
        "Tech Support",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

st.subheader(" Billing Information")

bill1, bill2, bill3 = st.columns(3)


with bill1:

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )


with bill2:

    paperless_billing = st.selectbox(
        "Paperless Billing",
        [
            "Yes",
            "No"
        ]
    )


with bill3:

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )


charges1, charges2 = st.columns(2)


with charges1:

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=200.0,
        value=70.0,
        step=0.01
    )


with charges2:

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        max_value=10000.0,
        value=840.0,
        step=0.01
    )


st.divider()


predict_button = st.button(
    "🔮 Predict Customer Churn",
    type="primary",
    use_container_width=True
)

if predict_button:


    customer = pd.DataFrame([{

        "gender": gender,

        "SeniorCitizen": senior_citizen,

        "Partner": partner,

        "Dependents": dependents,

        "tenure": tenure,

        "PhoneService": phone_service,

        "MultipleLines": multiple_lines,

        "InternetService": internet_service,

        "OnlineSecurity": online_security,

        "OnlineBackup": online_backup,

        "DeviceProtection": device_protection,

        "TechSupport": tech_support,

        "StreamingTV": streaming_tv,

        "StreamingMovies": streaming_movies,

        "Contract": contract,

        "PaperlessBilling": paperless_billing,

        "PaymentMethod": payment_method,

        "MonthlyCharges": monthly_charges,

        "TotalCharges": total_charges

    }])


    probability = model.predict_proba(
        customer
    )[0][1]


    THRESHOLD = 0.25


    if probability >= THRESHOLD:

        prediction = "CHURN"

    else:

        prediction = "NO CHURN"


    if probability >= 0.50:

        risk = "HIGH"

    elif probability >= 0.25:

        risk = "MEDIUM"

    else:

        risk = "LOW"


    st.divider()

    st.subheader(" Prediction Result")



    if risk == "HIGH":

        st.markdown(
            '<div class="risk-high">'
            ' HIGH CHURN RISK'
            '</div>',
            unsafe_allow_html=True
        )

    elif risk == "MEDIUM":

        st.markdown(
            '<div class="risk-medium">'
            '⚠️ MEDIUM CHURN RISK'
            '</div>',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            '<div class="risk-low">'
            '✅ LOW CHURN RISK'
            '</div>',
            unsafe_allow_html=True
        )


    st.write("")


    result1, result2, result3 = st.columns(3)


    with result1:

        st.metric(
            "Churn Probability",
            f"{probability:.2%}"
        )


    with result2:

        st.metric(
            "Prediction",
            prediction
        )


    with result3:

        st.metric(
            "Risk Level",
            risk
        )


    st.subheader("📈 Churn Probability")

    st.progress(
        float(probability)
    )

    st.caption(
        f"Model probability: {probability:.2%}"
    )


    st.subheader("💡 Recommended Action")


    if risk == "HIGH":

        st.error(
            """
            **Immediate retention action recommended.**

            Consider:
            - Personalized retention offer
            - Contract upgrade incentive
            - Dedicated customer support
            - Loyalty discount
            - Proactive customer outreach
            """
        )


    elif risk == "MEDIUM":

        st.warning(
            """
            **Targeted engagement recommended.**

            Consider:
            - Monitoring customer behavior
            - Personalized communication
            - Service improvement offers
            - Contract incentives
            """
        )


    else:

        st.success(
            """
            **Customer currently shows relatively low churn risk.**

            Continue normal engagement and monitor future
            customer behavior.
            """
        )


    st.subheader(" Decision Threshold")

    st.info(
        """
        The model uses a **0.25 decision threshold**.

        This threshold was selected through threshold optimization
        to improve churn detection recall.

        At the optimized threshold:

        **Precision ≈ 49.8% | Recall ≈ 81.0% | F1 ≈ 61.7%**
        """
    )


    st.subheader("📋 Customer Summary")

    summary1, summary2 = st.columns(2)


    with summary1:

        st.write(
            f"**Gender:** {gender}"
        )

        st.write(
            f"**Senior Citizen:** {senior_citizen}"
        )

        st.write(
            f"**Partner:** {partner}"
        )

        st.write(
            f"**Dependents:** {dependents}"
        )

        st.write(
            f"**Tenure:** {tenure} months"
        )


    with summary2:

        st.write(
            f"**Internet Service:** {internet_service}"
        )

        st.write(
            f"**Contract:** {contract}"
        )

        st.write(
            f"**Monthly Charges:** ${monthly_charges:.2f}"
        )

        st.write(
            f"**Total Charges:** ${total_charges:.2f}"
        )

        st.write(
            f"**Payment Method:** {payment_method}"
        )


st.divider()

st.caption(
    "Customer Churn Prediction | Machine Learning Portfolio Project"
)
