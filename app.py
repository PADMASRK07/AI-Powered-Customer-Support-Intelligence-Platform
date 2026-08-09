import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime

# --- Configuration & Styling ---
st.set_page_config(
    page_title="Support Intelligence Dashboard",
    page_icon="🤖",
    layout="wide"
)

# API Endpoint of your FastAPI backend
API_URL = "http://backend:8000/predict"

# Dictionary mapping model predictions to your strict text category ID values
TICKET_TYPE_MAP = {
    "LABEL_1" : 'Billing inquiry',
    "LABEL_2" : 'Cancellation request',
    "LABEL_3" : 'Product inquiry',
    "LABEL_4" : 'Refund request',
    "LABEL_5" : 'Technical issue'
}

# --- Title Header ---
st.title("🤖 AI-Powered Customer Support Intelligence Platform")
st.markdown("Automate text classification and prioritize tickets using fine-tuned NLP and Tabular architectures.")
st.divider()

# --- Main Layout Split ---
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.header("📋 New Ticket Triage")
    st.markdown("Enter customer and ticket details below for real-time model processing.")
    
    with st.form("ticket_form", clear_on_submit=False):
        # Customer Demographics fields
        st.subheader("👤 Customer Information")
        customer_name = st.text_input("Customer Name", placeholder="e.g., John Doe")
        
        c_age, c_gender = st.columns(2)
        with c_age:
            customer_age = st.number_input("Customer Age", min_value=1, max_value=120, value=1)
        with c_gender:
            customer_gender = st.selectbox("Customer Gender", ["Male", "Female", "Other"])
            
        customer_email = st.text_input("Customer Email", placeholder="e.g., john.doe@example.com")
        
        # Ticket & Product Specific Fields
        st.subheader("🎫 Ticket Details")
        ticket_subject = st.text_input("Ticket Subject", placeholder="e.g., Screen won't turn on")
        ticket_text = st.text_area(
            "Ticket Description",
            placeholder="Type or paste the raw customer issue message here...",
            height=120
        )
        
        # Comprehensive products listing
        product_purchased = st.selectbox(
            "Product Purchased",
            [
                "Amazon Echo", "Amazon Kindle", "Apple AirPods", "Asus ROG", "Autodesk AutoCAD", 
                "Bose QuietComfort", "Bose SoundLink", "Canon DSLR", "Canon EOS", "Dell XPS", 
                "Dyson Vacuum", "Fitbit Charge", "Fitbit Versa", "Garmin Forerunner", "Google Nest", 
                "Google Pixel", "GoPro Action Camera", "GoPro Hero", "HP Pavilion", "iPhone", 
                "Lenovo ThinkPad", "LG OLED TV", "LG Smart TV", "LG Washing Machine", "MacBook Pro", 
                "Microsoft Office", "Microsoft Surface", "Microsoft Windows", "Nest Thermostat", 
                "Nikon DSLR", "Nintendo Switch", "Philips Hue Lights", "PlayStation 5", "Roomba Robot Vacuum", 
                "Samsung Galaxy", "Samsung Smart TV", "Sony 4K TV", "Sony PlayStation", "Sony Xperia", "Xbox Series X"
            ]
        )
        
        date_of_purchase = st.date_input(
            "Date of Purchase",
            value=datetime.today()
        )
        
        # Form Submit Button using modern 2026 stretch width
        submitted = st.form_submit_button("Submit", width="stretch")

with col2:
    st.header("📊 Intelligence Engine Results")
    
    if submitted:
        if not ticket_text.strip() or not ticket_subject.strip():
            st.warning("⚠️ Both Ticket Subject and Ticket Description fields are required.")
        else:
            combined_text = f"{ticket_subject} {ticket_text}"
            days_open_calculated = max(0, (datetime.today().date() - date_of_purchase).days)
            
            payload = {
                "text": combined_text,
                "category": product_purchased,
                "user_tier": customer_gender,
                "days_open": int(days_open_calculated)
            }
            
            with st.spinner("Processing NLP Transformers & Tabular Rules..."):
                try:
                    response = requests.post(API_URL, json=payload, timeout=None)
                    
                    if response.status_code == 200:
                        result = response.json()
                        raw_category = result.get("ticket_category")
                        pred_priority = result.get("predicted_priority")
                        confidence = result.get("category_confidence")
                        
                        # Apply context strict mapping constraint
                        mapped_ticket_value = TICKET_TYPE_MAP.get(raw_category, "Unknown Category")
                        
                        st.success(f"✅ Inference Completed Successfully for {customer_name if customer_name else 'Customer'}")
                        
                        # Metrics Display layout
                        m_col1, m_col2 = st.columns(2)
                        with m_col1:
                            st.metric(label="Predicted Ticket Type ID", value=mapped_ticket_value)
                        with m_col2:
                            color_map = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}
                            indicator = color_map.get(pred_priority, "")
                            st.metric(label="Calculated Urgency Level", value=f"{indicator} {pred_priority}")
                        
                        st.markdown(f"**Raw Value Match:** `{raw_category}`")
                        st.markdown(f"**NLP Model Confidence Score: `{confidence * 100:.2f}%`**")
                        st.progress(float(confidence))
                        
                        st.subheader("🛠️ Prescriptive Action Summary")
                        if pred_priority in ["Critical", "High"]:
                            st.error(
                                f"🔥 **SLA Escalation Triggered**: This ticket requires immediate routing. "
                                f"Forwarding details for {customer_email} to the senior engineer team handling the **{product_purchased}** queue."
                            )
                        else:
                            st.info(
                                f"📋 **Standard Workflow Routing**: Logged normal priority queue requirements. "
                                f"Assigned to general tracking parameters for **{product_purchased}** support."
                            )
                            
                    else:
                        st.error(f"❌ Backend Server Error (Code {response.status_code}): {response.text}")
                        
                except requests.exceptions.ConnectionError:
                    st.error(
                        "❌ Connection Failed. Could not reach the FastAPI backend server. "
                        "Verify your FastAPI application is running locally on port 8000 using uvicorn."
                    )
    else:
        st.info("💡 Fill out the ticket parameters on the left and click submit to trigger model processing pipelines.")
