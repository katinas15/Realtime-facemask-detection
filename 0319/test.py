import pixellib
from pixellib.instance import instance_segmentation
import cv2

segment_video = instance_segmentation()
segment_video.load_model("C:/Users/laimo/Documents/Realtime-facemask-detection/0319/model.h5")
segment_video.process_video("C:/Users/laimo/Documents/Realtime-facemask-detection/0319/su_kauke1.mp4", 
    show_bboxes = True, 
    frames_per_second= 15, 
    output_video_name=r"su_kauke_output.mp4"
)