# =============================================================================
# XAVAD: Unified Explainable Anonymized Video Anomaly Detection (VAD) System
# =============================================================================
# This script provides Configs for XAVAD

# Author: Mulugeta W. Asres
# Email: muleina2000@gmail.com
# Date: October 2025
# =============================================================================
input_data_source_formats = ["webcam", "image", "video"]
input_imgsz = (320, 240) # w

seg_model_sel = "yolo"
# seg_model_sel = "rfdetr"
# seg_model_sel = "grounddino_sam"
od_model_dict = dict(yolo={ 
                            "instant": {"model":f"yolov8m-seg.pt", 
                                        "kwargs":{"detection_thr":0.15}}
                            },
                        rfdetr = {
                                "instant": {"model":"RFDETRSegMedium", 
                                            "kwargs":{"detection_thr":0.25}},
                                },
                        grounddino_sam = {
                                        "instant": {"model":["grounding-dino-tiny", "sam2.1-hiera-large"], 
                                                    "kwargs":{"detection_thr":0.35}} 
                                        }
                    )

# From the COCO Dataset, let"s create a list of classes that Seg Model could be able to detect
od_classnames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
            "potted plant", "bed", "dining table", "toilet", "tv monitor", "laptop", "mouse", "remote", "keyboard",
            "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
            "teddy bear", "hair drier", "toothbrush"]

od_classid_mapper_dict = {k: i for i, k in enumerate(od_classnames)}
od_classname_list = ["person"]
# od_imgsz = (320, 240)
od_imgsz = (640, 640)
# od_imgsz = None
od_color_format = "bgr"
od_name = f"body__{seg_model_sel}"
od_version = od_model_dict[seg_model_sel]["instant"]["model"]
od_thr = od_model_dict[seg_model_sel]["instant"]["kwargs"]["detection_thr"]
kwargs_detector_method = {"seg_type": "instant", 
                          "imgsz": od_imgsz, 
                          "detector_version": od_version, 
                          "detection_thr":od_thr, 
                          "classes":[od_classid_mapper_dict[item] for item in od_classname_list]
                          }
anny_transform_dict = {
                        "RAW_IMAGE": dict(anony_method="raw", 
                                    kwargs_anony_method={"color_format":od_color_format}, 
                                    detector_name="image", 
                                    kwargs_detector_method=kwargs_detector_method
                                    ),
                        f"MASKED": dict(anony_method="silhoutte",    
                                        kwargs_anony_method={"color_format":od_color_format, "alpha": 1.0, "color":"#000000"}, 
                                        detector_name=od_name, 
                                        kwargs_detector_method=kwargs_detector_method
                                        ),
                        f"EDGED": dict(anony_method="edge", 
                                    kwargs_anony_method={"color_format":od_color_format, "thr1": 100, "thr2":200}, 
                                    detector_name=od_name, 
                                    kwargs_detector_method=kwargs_detector_method
                                    ),
                        "BLURRED_NA": dict(anony_method = "guassian_blur", 
                                    kwargs_anony_method={"color_format":od_color_format, "kernelsize": (13, 13), "sigma": (10,10), "use_adaptive_mask":False}, 
                                    detector_name=od_name, 
                                    kwargs_detector_method=kwargs_detector_method,  
                                ),
                        "BLURRED_L_A": dict(anony_method = "guassian_blur", 
                                    kwargs_anony_method={"color_format":od_color_format, "kernelsize": (13, 13), "sigma":(10,10), "use_adaptive_mask":True, "adaptive_sigma":False, "alpha_mask_scale":1.0, "alpha_dim_scale":0.5, "use_log_scaler":True}, 
                                    detector_name=od_name, 
                                    kwargs_detector_method=kwargs_detector_method,  
                                ),
                        "BLURRED_L_A_FULL": dict(anony_method = "guassian_blur", 
                                    kwargs_anony_method={"color_format":od_color_format, "kernelsize": (13, 13), "sigma":(10,10), "use_adaptive_mask":True, "adaptive_sigma":True, "alpha_mask_scale":1.0, "alpha_dim_scale":0.5, "use_log_scaler":True}, 
                                    detector_name=od_name, 
                                    kwargs_detector_method=kwargs_detector_method,  
                                ),
                        "BLURRED_L_A_MAX": dict(anony_method = "guassian_blur", 
                                    kwargs_anony_method={"color_format":od_color_format, "kernelsize": (0, 0), "sigma":(0,0), "use_adaptive_mask":True}, 
                                    detector_name=od_name, 
                                    kwargs_detector_method=kwargs_detector_method,  
                                ),
                        "PIXELIZED_NA": dict(anony_method="pixelization",    
                                    kwargs_anony_method={"color_format":od_color_format, "downsize": 4, "use_adaptive_mask":False}, 
                                    detector_name=od_name, 
                                    kwargs_detector_method=kwargs_detector_method,  
                                    ),
                        "PIXELIZED_L_A": dict(anony_method="pixelization",    
                                    kwargs_anony_method={"color_format":od_color_format, "downsize": 4, "use_adaptive_mask":True, "alpha_mask_scale":1.0, "alpha_dim_scale":0.5, "use_log_scaler":True}, 
                                    detector_name=od_name, 
                                    kwargs_detector_method=kwargs_detector_method,  
                                    ),
                        "PIXELIZED_L_A_MAX": dict(anony_method="pixelization",    
                                        kwargs_anony_method={"color_format":od_color_format, "downsize": 0, "use_adaptive_mask":True}, 
                                        detector_name=od_name, 
                                        kwargs_detector_method=kwargs_detector_method,  
                                        )
                        }
anony_method_name_mapper_dict = {
                                "no-an":"RAW_IMAGE",
                                "mask":"MASKED", 
                                "edge":"EDGED", 
                                "blur":"BLURRED_NA", 
                                "adaptive_blur":"BLURRED_L_A", 
                                "adaptive_full_blur":"BLURRED_L_A_FULL", 
                                "adaptive_max_blur":"BLURRED_L_A_MAX", 
                                "pixelization":"PIXELIZED_NA",
                                "adaptive_pixelization":"PIXELIZED_L_A",
                                "adaptive_max_pixelization":"PIXELIZED_L_A_MAX",
                            }
anony_method_options_list = list(anony_method_name_mapper_dict.keys())
preprocessors_list = []
postprocessors_list = []

result_path = "results/"
