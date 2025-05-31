import streamlit as st
import razorpay
import os
import json
from datetime import datetime
import time

# --- Configuration ---
# Load keys from Streamlit secrets (recommended)
try:
    RAZORPAY_KEY_ID = st.secrets["RAZORPAY_KEY_ID"]
    RAZORPAY_KEY_SECRET = st.secrets["RAZORPAY_KEY_SECRET"]
except KeyError:
    st.error("ERROR: Razorpay API keys not found in Streamlit secrets.")
    st.info("Please create a `.streamlit/secrets.toml` file with your Test Keys:\n"
            "RAZORPAY_KEY_ID = \"rzp_test_YOUR_KEY_ID\"\n"
            "RAZORPAY_KEY_SECRET = \"YOUR_KEY_SECRET\"")
    st.stop()
except FileNotFoundError:
     st.error("ERROR: Streamlit secrets file (`.streamlit/secrets.toml`) not found.")
     st.info("Ensure the secrets file exists in the `.streamlit` directory.")
     st.stop()


# --- Initialize Razorpay Client ---
client = None
try:
    client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))
    client.set_app_details({"title" : "Streamlit Razorpay Test App", "version" : "1.0"})
except Exception as e:
    st.error(f"Failed to initialize Razorpay client: {e}")
    st.stop()

# --- Streamlit UI ---
st.set_page_config(page_title="Razorpay Test Payment", layout="centered")
st.title("🧪 Razorpay Payment Gateway Tester")
st.markdown("---")
st.warning("Ensure you are using **Test Mode** API Keys from Razorpay.", icon="⚠️")

# --- Input Payment Details ---
st.subheader("Payment Details")
amount_inr = st.number_input("Enter Amount (INR)", min_value=1.0, value=10.0, step=1.0, format="%.2f")
customer_name = st.text_input("Customer Name (Optional)", "Test Customer")
customer_email = st.text_input("Customer Email (Optional)", "test@example.com")
customer_contact = st.text_input("Customer Contact (Optional)", "+919999999999")

amount_paise = int(amount_inr * 100) # Convert to paise

# --- Payment Button and Logic ---
st.markdown("---")
pay_button = st.button("Proceed to Pay")

# Placeholder for the Razorpay button HTML
payment_button_placeholder = st.empty()
status_placeholder = st.empty()


if pay_button and client:
    # Basic validation
    if amount_paise <= 0:
        status_placeholder.error("Amount must be greater than 0.")
    else:
        status_placeholder.info("Creating Razorpay Order...")
        time.sleep(0.5) # Small delay for UX

        # 1. Create Razorpay Order
        receipt_id = f"streamlit_test_{int(datetime.now().timestamp())}"
        order_data = {
            "amount": amount_paise,
            "currency": "INR",
            "receipt": receipt_id,
            "notes": {
                "name": customer_name,
                "email": customer_email,
                "contact": customer_contact,
                "source": "Streamlit Test App"
            }
        }

        try:
            order_response = client.order.create(data=order_data)
            order_id = order_response['id']
            status_placeholder.success(f"Razorpay Order Created! Order ID: `{order_id}`")
            st.session_state['razorpay_order_id'] = order_id # Store order_id

            # 2. Prepare Checkout Options (remains the same)
            checkout_options = {
                "key": RAZORPAY_KEY_ID,
                "amount": amount_paise,
                "currency": "INR",
                "name": "Streamlit Test Co.",
                "description": f"Test Transaction for {amount_inr} INR",
                "image": "https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.svg",
                "order_id": order_id,
                "prefill": {
                    "name": customer_name,
                    "email": customer_email,
                    "contact": customer_contact
                },
                "notes": {
                    "address": "Test Address, Streamlit City"
                },
                "theme": {
                    "color": "#3399cc"
                },
                 "modal": {
                    "ondismiss": """function(){
                        console.log('Checkout form closed');
                    }"""
                 }
            }
            options_json = json.dumps(checkout_options)


            # 3. Create HTML/JS for the Razorpay Button - MODIFIED STYLES
            button_id = f"rzp-button-{order_id}"

            html_code = f"""
            <html >
            <head>
            <script src="https://checkout.razorpay.com/v1/checkout.js"></script>
            <style>
              /* Optional: Center the button within its container */
              body {{ display: flex; justify-content: center; align-items: center; height: 100%; margin: 0; }}
            </style>
            </head>
            <body>
            <button
                id="{button_id}"
                style="
                    padding: 18px 35px;       /* Increased padding */
                    font-size: 22px;          /* Increased font size */
                    color: white;
                    background-color: #FF4B4B; /* Changed color for more pop */
                    border: none;
                    border-radius: 8px;       /* Slightly more rounded */
                    cursor: pointer;
                    font-weight: bold;        /* Make text bolder */
                    width: auto;              /* Allow button to size naturally or set to '80%' or similar */
                    text-align: center;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.2); /* Add subtle shadow */
                    transition: background-color 0.2s ease; /* Smooth hover effect */
                "
                onmouseover="this.style.backgroundColor='#E03A3A';" /* Darken on hover */
                onmouseout="this.style.backgroundColor='#FF4B4B';" /* Revert on mouse out */
            >
                Pay ₹{amount_inr:.2f} Now
            </button>

            <script>
            var options = {options_json};

             options.modal = {{
                 ondismiss: function(){{
                    console.log('Checkout form was closed by the user.');
                 }}
             }};

            var rzp1 = new Razorpay(options);

            document.getElementById('{button_id}').onclick = function(e){{
                console.log("Razorpay button clicked, opening checkout...");
                try {{
                    rzp1.open();
                }} catch (error) {{
                    console.error("Error opening Razorpay checkout:", error);
                    alert("Error opening payment gateway. See console for details.");
                }}
                e.preventDefault();
            }}
            </script>
            </body>
            </html>
            """

            # 4. Display the button using st.components.v1.html - Adjust height if needed
            payment_button_placeholder.markdown("👇 Click the button below to open Razorpay Checkout 👇")
            # Increased height slightly to accommodate the larger button
            st.components.v1.html(html_code, height=120)
            status_placeholder.success(f"Razorpay Order Created! Order ID: `{order_id}`. Click the button above to pay.")
            st.info(f"**After payment attempt, please manually verify the status for Order ID `{order_id}` in your [Razorpay Test Dashboard](https://dashboard.razorpay.com/app/orders).**", icon="ℹ️")



        except razorpay.errors.BadRequestError as e:
            status_placeholder.error(f"Razorpay API Error (Bad Request): {e}")
            st.write(e.description) # More specific error details if available
        except Exception as e:
            status_placeholder.error(f"An unexpected error occurred: {e}")

# --- Display Order ID if it exists in session state ---
if 'razorpay_order_id' in st.session_state:
     st.markdown("---")
     st.write(f"Last generated Order ID: `{st.session_state['razorpay_order_id']}`")
     st.markdown("Remember to check its status in the Razorpay Test Dashboard.")