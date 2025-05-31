import razorpay
import logging
from dotenv import load_dotenv

import os
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
try:
    razorpay_client = razorpay.Client(auth=(os.getenv("RAZORPAY_KEY_ID"), os.getenv("RAZORPAY_KEY_SECRET")))

    plans = razorpay_client.plan.all()
    logger.info(f"Plans: {plans}")
except Exception as e:
    logger.error(f"Failed to fetch plans: {str(e)}")