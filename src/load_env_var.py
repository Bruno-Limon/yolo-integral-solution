import os

def load_var(iothub):
    if iothub == False:
        from dotenv import load_dotenv
        load_dotenv()

    list_var_names = ['VIDEO_SOURCE', 'DO_SKIP_FRAMES', 'SHOW_IMAGE', 'SAVE_VIDEO', 'CONNECTION_STRING',
                      'EXPOSE_STREAM', 'RUN_WAIT_TIME', 'FLASK_PORT', 'IS_STREAM',
                      'DEVICE_ID', 'CAMERA_ID', 'MODEL_ID',
                      'DO_DRAW_BBOX', 'DO_MAN_DOWN', 'DO_DRAW_TRACKS', 'DO_TIME_ZONE', 'DO_COUNT_OBJECTS', 'DO_COUNT_ZONE', 'DO_ENTER_LEAVE',
                      'SHOW_MAN_DOWN', 'SHOW_ZONE', 'SHOW_COUNT_PEOPLE', 'SHOW_TIME_ZONE', 'SHOW_ENTER_LEAVE',
                      'ZONE_COORDS', 'DOOR_COORDS', 'DOOR1_COORDS', 'DOOR2_COORDS']
    dict_var = {}

    for var_name in list_var_names:
        dict_var[var_name] = os.getenv(key=var_name, default='variable not found in .env file')

    return dict_var

if __name__ == "__main__":
    dict_var = load_var(iothub=False)
    for key, val in dict_var.items():
        print(f"{key} : {val}")