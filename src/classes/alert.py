import time
import logging
import os

from config import enable_local_work
enable_local_work()

ENV_VAR_TRUE_LABEL = "true"
ALERT_LIST = ['man down', 'people in zone']

class Alert:
    def __init__(self,
                 alert_type,
                 timeout_minutes=os.environ['ALERT_TIMEOUT']):

        self.timeout = float(timeout_minutes)*60
        if alert_type in ALERT_LIST:
            self.alert_type=alert_type
        else:
            raise ValueError(f"Error: 'alert_type' must be in {ALERT_LIST}")

        self.start_timeout = None
        self.history = []
        self.elapsed = []


    def _frames_true(self):
        """
        Tell you the number of consecutive TRUE frames.
        So, in case of man down, it can tell you the number of consecutive
        frames a person is down.
        """
        count = 0
        # Traverse the array from the end
        for i in range(len(self.history) - 1, -1, -1):
            if self.history[i]:
                count += 1
            else:
                break  # Stop counting when a False is encountered
        return count

    def is_man_down(self):
        """
        Check if a person is down for more than N consecutive seconds.
        """
        n_frames = self._frames_true()
        if n_frames <= 0:
            return False

        time_man_down = sum(self.elapsed[-n_frames:])
        if time_man_down > float(os.environ['ALERT_TIME_MAN_DOWN']):
            return True

        return False

    def is_too_many_people(self):
        """
        Check if there are too many people in zone for more than N consecutive seconds.
        """
        n_frames = self._frames_true()
        if n_frames <= 0:
            return False

        time_man_down = sum(self.elapsed[-n_frames:])
        if time_man_down > int(os.environ['ALERT_MAX_PEOPLE_ZONE']):
            return True

        return False

    def _reset_history(self, total_reset=False):
        n_frame_man_down = self._frames_true()
        if (n_frame_man_down > 0) and n_frame_man_down < len(self.history) and (not total_reset):
            self.history = self.history[-n_frame_man_down:]
            self.elapsed = self.elapsed[-n_frame_man_down:]
        else:
            self.history = []
            self.elapsed = []

    def send_message_iot_hub(self, message:str):
        logging.warn(message)
        return message

    def update_man_down(self, is_man_down:bool, elapsed:float):
        """
        If true an alert is sent
        """

        if sum(self.elapsed) > self.timeout:
            self._reset_history()

        self.history.append(is_man_down)
        self.elapsed.append(elapsed)
        logging.debug(f"man_down {self.history}")
        logging.debug(f"elapsed {self.elapsed}")

        n_frames = self._frames_true()
        if n_frames <= 0:
            return False

        time_man_down = sum(self.elapsed[-n_frames:])
        logging.debug(f"time Man Down {time_man_down}")

        if self.is_man_down():
            if self.start_timeout is None:
                self.start_timeout = time.time()
                self.send_message_iot_hub("IOTHUB: Man Down")
                self._reset_history(total_reset=True)
                return True

            actual_time = time.time()
            if (actual_time-self.start_timeout) >= self.timeout:
                self.start_timeout = actual_time
                self.send_message_iot_hub("IOTHUB: Man Down")
                self._reset_history(total_reset=True)
                return True

            return False

        return False

    def update_people_in_zone(self, is_too_many_people_in_zone:bool, elapsed:float):
        """
        If true an alert is sent
        """
        if sum(self.elapsed) > self.timeout:
            self._reset_history()

        self.history.append(is_too_many_people_in_zone)
        self.elapsed.append(elapsed)
        logging.debug(f"too_many_people {self.history}")
        logging.debug(f"elapsed {self.elapsed}")

        n_frames = self._frames_true()
        if n_frames <= 0:
            return False

        time_too_many_people = sum(self.elapsed[-n_frames:])
        logging.debug(f"time Too Many People: {time_too_many_people}")

        if self.is_too_many_people():
            if self.start_timeout is None:
                self.start_timeout = time.time()
                self.send_message_iot_hub("IOTHUB: Too many people")
                self._reset_history(total_reset=True)
                return True

            actual_time = time.time()
            if (actual_time-self.start_timeout) >= self.timeout:
                self.start_timeout = actual_time
                self.send_message_iot_hub("IOTHUB: Too many people")
                self._reset_history(total_reset=True)
                return True
            return False
        return False


    def update(self, history:bool, elapsed:float):

        if self.alert_type == 'man down':
            alert_sent = self.update_man_down(history, elapsed)
        elif self.alert_type == 'people in zone':
            alert_sent = self.update_people_in_zone(history, elapsed)
        else:
            raise ValueError(f"Error: 'alert_type' must be in {ALERT_LIST}.")

        return alert_sent, self.alert_type

