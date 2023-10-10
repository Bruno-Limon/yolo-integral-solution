import time
import logging
import smtplib
from email.mime.text import MIMEText
import os
import config

if config.env_vars_local == True:
    from dotenv import load_dotenv
    load_dotenv()

ENV_VAR_TRUE_LABEL = "true"


class Alert:
    def __init__(self,
                 timeout_minutes=os.environ['ALERT_TIMEOUT'],
                 mail=os.environ['ALERT_SEND_MAIL'],
                 sender=os.environ['ALERT_SENDER'],
                 password=os.environ['ALERT_PASSWORD'],
                 subject="Alert",
                 mail_body="Test alert mail from Insiel",
                 recipients=os.environ['ALERT_RECIPIENTS']
                 ):

        self.timeout = float(timeout_minutes)*60
        self.start_timeout = None
        self.mail = mail
        self.sender = sender
        self.password = password
        self.subject = subject
        self.body = mail_body
        self.recipients = [x for x in recipients.split(',')]
        self.man_down_history = []
        self.man_down_elapsed = []

    def send_email(self):
        """
        Send an email using the SMTP protocol with SSL encryption.

        Raises:
        - smtplib.SMTPException: If an error occurs during the email sending process.
        """
        # Create a MIMEText message with the provided body.
        msg = MIMEText(self.body)

        # Set the email header fields.
        msg['Subject'] = self.subject
        msg['From'] = self.sender
        msg['To'] = ', '.join(self.recipients)

        # Connect to the SMTP server with SSL encryption.
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp_server:
            # Login to the SMTP server using the sender's email address and password.
            smtp_server.login(self.sender, self.password)

            # Send the email to the recipients.
            smtp_server.sendmail(self.sender, self.recipients, msg.as_string())

        # Print a success message after sending the email.
        print("Message sent!")

    def _frames_down(self):
        """
        Tell you the number of consecutive frames a person is down.
        """
        count = 0
        # Traverse the array from the end
        for i in range(len(self.man_down_history) - 1, -1, -1):
            if self.man_down_history[i]:
                count += 1
            else:
                break  # Stop counting when a False is encountered
        return count

    def is_man_down(self):
        """
        Check if a person is down for more than N consecutive seconds.
        """
        n_frames = self._frames_down()
        if n_frames <= 0:
            return False

        time_man_down = sum(self.man_down_elapsed[-n_frames:])
        if time_man_down > float(os.environ['ALERT_TIME_MAN_DOWN']):
            return True

        return False

    def _reset_history(self, total_reset=False):
        n_frame_man_down = self._frames_down()
        if (n_frame_man_down > 0) and n_frame_man_down < len(self.man_down_history) and (not total_reset):
            self.man_down_history = self.man_down_history[-n_frame_man_down:]
            self.man_down_elapsed = self.man_down_elapsed[-n_frame_man_down:]
        else:
            self.man_down_history = []
            self.man_down_elapsed = []


    def update_man_down(self, is_man_down:bool, elapsed:float):
        """
        If true an allert has been sended.
        """
        if sum(self.man_down_elapsed) > self.timeout:
            self._reset_history()

        self.man_down_history.append(is_man_down)
        self.man_down_elapsed.append(elapsed)
        # logging.info(f"man_down {self.man_down_history}")
        # logging.info(f"elapsed {self.man_down_elapsed}")

        n_frames = self._frames_down()
        if n_frames <= 0:
            return False

        time_man_down = sum(self.man_down_elapsed[-n_frames:])
        # logging.info(f"time Man Down {time_man_down}")

        if self.is_man_down():
            if self.start_timeout is None:
                self.start_timeout = time.time()

                if self.mail == ENV_VAR_TRUE_LABEL:
                    self.send_email
                    self._reset_history(total_reset=True)

                return True

            actual_time = time.time()
            if (actual_time-self.start_timeout) >= self.timeout:
                self.start_timeout = actual_time

                if self.mail == ENV_VAR_TRUE_LABEL:
                    self.send_email()
                    self._reset_history(total_reset=True)

                return True

            return False

        return False


