# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import asyncio
import random
import uuid

import numpy as np

# Using the Python Device SDK for IoT Hub:
#   https://github.com/Azure/azure-iot-sdk-python
#   Run 'pip install azure-iot-device' to install the required libraries for this application
#   Note: Requires Python 3.6+

# The sample connects to a device-specific MQTT endpoint on your IoT Hub.
from azure.iot.device.aio import IoTHubDeviceClient
from azure.iot.device import Message

from yolodetect import detect

if __name__ == '__main__':
    print("IoT Hub simulated device")
    print("Press Ctrl-C to exit")

    vid_path = '../Data/vid5.mp4'

    # zone to count people in
    zone_poly = np.array([[460, 570], #x1, y1 = left upper corner
                        [1270, 570],#x2, y2 = right upper corner
                        [1265, 710],#x3, y3 = right lower corner
                        [430, 710]], np.int32) #x4, y4 = left lower corner
    zone_poly = zone_poly.reshape((-1, 1, 2))

    # calling main detection function, passing all necessary arguments
    detect(vid_path=vid_path,
        zone_poly=zone_poly,
        do_man_down=False,
        show_keypoints=True,
        show_down_onscreen=True,
        do_count_objs=True,
        show_count_onscreen=True,
        show_box=True,
        do_count_zone=True,
        show_zone_onscreen=True,
        print_obj_info=True,
        save_video=False)
