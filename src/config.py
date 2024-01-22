"""
config file to set variables, namely:
    - load environmente variables in case of working in local version, using
      the variable "local_work=True", if deploying to edge device set to False
    - configure the tracker yaml file to set variables depending on the values of
      environment variables
"""
import os

LOCAL_WORK = True

def enable_local_work():
    if LOCAL_WORK:
        from dotenv import load_dotenv
        load_dotenv()

def write_yaml(file_name, list_parameters):
    with open(file_name, 'w') as f:
        f.write("# Default YOLO tracker settings for BoT-SORT " +
                "and Byte-Track trackers https://github.com/NirAharon/BoT-SORT " +
                "\n")

    for var in list_parameters:
        val = os.environ[var]
        with open(file_name, 'a') as f:
            f.write(f"{var}: {val}\n")

def config_tracker():
    list_parameters = ['tracker_type', 'track_high_thresh', 'track_low_thresh',
                       'new_track_thresh', 'track_buffer', 'match_thresh']
    file_name = "src/tracker.yaml"

    # if file is already found, delete and recreate
    if os.path.isfile(file_name):
        os.remove(file_name)
        write_yaml(file_name, list_parameters)

    # if file not found, create it
    else:
        write_yaml(file_name, list_parameters)
