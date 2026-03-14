import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging

logger = logging.getLogger(__name__)

# Standard SMTP configuration pulled from .env for completely Real Email capability without API costs
async def send_otp_email(to_email: str, otp: str, username: str = None):
    """
    Sends an OTP code via Real SMTP email. 
    If you haven't supplied 'SMTP_EMAIL' in your environment,
    it automatically degrades gracefully to print the OTP in your Dev Console.
    """
    SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
    SMTP_EMAIL = os.getenv("SMTP_EMAIL", "") 
    SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")

    # Validation
    if not SMTP_EMAIL or not SMTP_PASSWORD:
        # Graceful fallback to console intercept if user hasn't configured SMTP yet
        print("\n" + "="*50)
        print(" 📧  NOVA EMAIL SERVICE (MOCK INTERCEPT)")
        print(" ⚠️   To get REAL emails, update SMTP_EMAIL in your .env")
        print("="*50)
        print(f" To:      {to_email}")
        if username:
            print(f" Subject: Welcome to NovaOS, {username}!")
        else:
            print(f" Subject: Your NovaOS Login Code")
        print(f" ")
        print(f" Your NovaOS Temporary Access Code is:")
        print(f" ")
        print(f" >>  {otp}  <<")
        print(f" ")
        print(f" This code expires in 10 minutes.")
        print("="*50 + "\n")
        logger.info(f"Mock email sent to {to_email} with OTP {otp}.")
        return True

    # ---- REAL SMTP DELIVERY ----
    try:
        msg = MIMEMultipart()
        msg['From'] = SMTP_EMAIL
        msg['To'] = to_email
        if username:
            msg['Subject'] = f"Welcome to NovaOS, {username}!"
        else:
            msg['Subject'] = "Your NovaOS Login Code"

        body = f"Your NovaOS Temporary Access Code is:\n\n >>  {otp}  <<\n\nThis code expires in 10 minutes."
        msg.attach(MIMEText(body, 'plain', 'utf-8'))

        # Connect, Secure, and Dispatch
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_EMAIL, SMTP_PASSWORD)
        server.send_message(msg)
        server.quit()

        logger.info(f"REAL SMTP email OTP successfully dispatched to {to_email}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to push real SMTP email: {str(e)}")
        # Don't completely fail local tests if they fed a bad password
        return False
