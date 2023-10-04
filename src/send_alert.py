import smtplib, requests, time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

sender_gmail_id = ""
sender_gmail_pw = ""
receiver_gmail_id = ""

# creates SMTP session
s = smtplib.SMTP("smtp.gmail.com", 587)
# start TLS for security
s.starttls()
# Authentication
s.login(sender_gmail_id, sender_gmail_pw)

# Instance of MIMEMultipart
msg = MIMEMultipart("alternative")
msg["Subject"]= "man_down test"
msg["From"] = sender_gmail_id
msg["To"] = receiver_gmail_id

# Plain text body of the mail
text = "content of mail"

# Attach the Plain body with the msg instance
msg.attach(MIMEText(text, "plain"))

# # HTML body of the mail
# html ="<h2>Your site is running now.</h2><br/><a href ='" + "'>Click here to visit.</a>"
# # Attach the HTML body with the msg instance
# msg.attach(MIMEText(html, "html"))

# Sending the mail
s.sendmail(sender_gmail_id, receiver_gmail_id, msg.as_string())
s.quit()
print('sent')