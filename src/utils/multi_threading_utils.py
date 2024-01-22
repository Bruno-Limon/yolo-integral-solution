import cv2
from utils.detection_utils import compute_detection, compute_postprocessing
from utils.postprocessing_utils import aggregate_info
from concurrent.futures import ThreadPoolExecutor


def run_reading(video_path, queue):
    """
    TODO
    """
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()

    while success:
        success, frame = cap.read()
        queue.put(cap)
        queue.put(frame)

def run_processing(model_detection, model_pose, frame, cap, init_vars, queue):
    """
    TODO
    """
    labels_dict = init_vars['labels_dict']

    list_objects, pose_results, infer_time = compute_detection(model_detection,
                                                               model_pose,
                                                               frame,
                                                               labels_dict)

    results_postproc, post_process_time, frame = compute_postprocessing(list_objects,
                                                                 pose_results,
                                                                 frame,
                                                                 init_vars,
                                                                 cap)

    queue.put(results_postproc)
    queue.put(list_objects)

    # while frame:
    #     cv2.imshow("frame1", frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

def run_message(list_aggregates, results_postprocessing, list_objects):
    """
    TODO
    """

    list_aggregates = aggregate_info(list_aggregates,
                                     results_postprocessing,
                                     list_objects)





















