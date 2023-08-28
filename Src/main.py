# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.
# Using the Python Device SDK for IoT Hub:
#   https://github.com/Azure/azure-iot-sdk-python
#   Run 'pip install azure-iot-device' to install the required libraries for this application
#   Note: Requires Python 3.6+

# The sample connects to a device-specific MQTT endpoint on your IoT Hub.
import asyncio
import uuid
import numpy as np

from azure.iot.device.aio import IoTHubDeviceClient
from azure.iot.device import Message

from yolodetect import detect

# The device connection string to authenticate the device with your IoT hub.
CONNECTION_STRING = "HostName=ioh-innovideo.azure-devices.net;DeviceId=local_bruno;SharedAccessKey=WRg0acdJGJnizgXv5y/+zG0I8pEFIbwtCH63W45YIrs="
MESSAGE_TIMEOUT = 10

# Define the message to send to IoT Hub.

# Temperature threshold for alerting
TEMP_ALERT_THRESHOLD = 30

async def main():
    # video source
    vid_path = '../Data/vid2.mp4'

    # zone to count people in
    zone_poly = np.array([[460, 570],  #x1, y1 = left upper corner
                          [1270, 570], #x2, y2 = right upper corner
                          [1265, 710], #x3, y3 = right lower corner
                          [430, 710]], np.int32) #x4, y4 = left lower corner
    zone_poly = zone_poly.reshape((-1, 1, 2))

    for list_obj_info in detect(vid_path=vid_path,
                                show_image=True,
                                zone_poly=zone_poly,
                                do_man_down=True,
                                show_keypoints=True,
                                show_down_onscreen=True,
                                do_count_objs=True,
                                show_count_onscreen=True,
                                show_box=True,
                                do_count_zone=True,
                                show_zone_onscreen=True,
                                save_video=False):

        for x in list_obj_info:
            try:
                client = IoTHubDeviceClient.create_from_connection_string(CONNECTION_STRING)
                await client.connect()

                print("IoT Hub device sending periodic messages, press Ctrl-C to exit")

                while True:
                    # Build the message with simulated telemetry values.
                    message = Message(x)

                    # Add standard message properties
                    message.message_id = uuid.uuid4()
                    message.content_encoding = "utf-8"
                    message.content_type = "application/json"

                    # Send the message.
                    print("Sending message: %s" % message.data)
                    try:
                        await client.send_message(message)
                    except Exception as ex:
                        print("Error sending message from device: {}".format(ex))
                    await asyncio.sleep(1)

            except Exception as iothub_error:
                print("Unexpected error %s from IoTHub" % iothub_error)
                return
            except asyncio.CancelledError:
                await client.shutdown()
                print('Shutting down device client')

if __name__ == '__main__':
    print("IoT Hub simulated device")
    print("Press Ctrl-C to exit")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('Keyboard Interrupt - sample stopped')
