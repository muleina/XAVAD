# =============================================================================
# XAVAD: Unified Explainable Anonymized Video Anomaly Detection (VAD) System
# =============================================================================
# This script provides Object Detection and Segmentation functions for XAVAD

# Author: Mulugeta W. Asres
# Email: muleina2000@gmail.com
# Date: October 2025
# =============================================================================

import os
import torch
import numpy as np
import cv2
import copy
from typing import List, Tuple, Dict, Union, Optional, Literal, Any
import utilities as util
from config import config as cfg

src_path = os.path.abspath(os.path.dirname(__file__))
current_path = os.path.abspath(os.path.dirname(__file__))

class RFDETR:
    
    def __init__(self, modelname:str="RFDETRSegMedium", device: Literal["cuda", "cpu"] = "cuda", **kwargs):

        # Choose your size (Nano, Small, Medium, etc.)
        from rfdetr.util.coco_classes import COCO_CLASSES
        self.idx2classname_mapper = COCO_CLASSES
        self.classname2idx_mapper = dict({v:k for k,v in COCO_CLASSES.items()})
        self.modelname = modelname
        self.device = device if torch.cuda.is_available() else "cpu"
        
        print(f"loading {modelname}")
        
        # Load the model
        if self.modelname == "RFDETRSegNano":
            from rfdetr import RFDETRSegNano
            model = RFDETRSegNano(device=self.device)
        elif self.modelname == "RFDETRSegSmall":
            from rfdetr import RFDETRSegSmall
            model = RFDETRSegSmall(device=self.device)
        elif self.modelname == "RFDETRSegMedium":
            from rfdetr import RFDETRSegMedium
            model = RFDETRSegMedium(device=self.device)
        elif self.modelname == "RFDETRSegLarge":
            from rfdetr import RFDETRSegLarge
            model = RFDETRSegLarge(device=self.device)
        elif self.modelname == "RFDETRSegXLarge":
            from rfdetr import RFDETRSegXLarge
            model = RFDETRSegXLarge(device=self.device)
        elif self.modelname == "RFDETRSeg2XLarge":
            from rfdetr import RFDETRSeg2XLarge
            model = RFDETRSeg2XLarge(device=self.device)
        else:
            raise NotImplementedError
        model.optimize_for_inference()
        self.model = model
         
    def predict(self, x: util.PIL.Image.Image | np.ndarray, classes: List[str] = ["person"], **kwargs):
        # print(kwargs)
        results = self.model.predict(x, 
                                    threshold=kwargs.get("threshold", 0.5),  
                                    shape=kwargs.get("shape", None),  
                                    # shape=(640, 640) # diom must be 24*
                                    # quantization=kwargs.get("quantization", None)#="fp16"
                                    )
        
        if results.mask is None:
            return None
        
        if classes is not None:
            sel_classes = [self.idx2classname_mapper[class_id] in classes for class_id in results.class_id]
            if len(sel_classes):
                masks = np.array(results[sel_classes].mask)
            else:
                return None
        else:
            np.array(results.mask)
            
        if len(masks):
            return masks
        else:
            return None
        # labels = [f"{COCO_CLASSES[class_id]}" for class_id in detections.class_id]
        # annotated_image = sv.MaskAnnotator().annotate(detections.metadata["source_image"], detections)
        # # util.PIL.Image.fromarray(annotated_image)
        # annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)
        # util.PIL.Image.fromarray(annotated_image)
    
class GroundDinoSAMAN:
    
    def __init__(self, grounddino_modelname: str = "grounding-dino-tiny", sam_modelname: Literal["sam2.1-hiera-tiny", "sam2.1-hiera-small", "sam2.1-hiera-large"] = "sam2.1-hiera-large", device: Literal["cuda", "cpu"] = "cuda", **kwargs):
        # using negative processor
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        from transformers import Sam2Processor, Sam2Model

        print(f"loading {grounddino_modelname}, and {sam_modelname}")
        self.grounddino_modelname = grounddino_modelname
        self.sam_modelname = sam_modelname
        self.device = device if torch.cuda.is_available() else "cpu"

        # Detect the "person" using a zero-shot detector
        detector_id = f"IDEA-Research/{self.grounddino_modelname}"
        self.detector_processor = AutoProcessor.from_pretrained(detector_id)
        self.detector_model = AutoModelForZeroShotObjectDetection.from_pretrained(detector_id).to(self.device)

        # Pass those bounding boxes into SAM 2 to get pixel-perfect masks
        sam_checkpoint = f"facebook/{self.sam_modelname}"

        self.sam_processor = Sam2Processor.from_pretrained(sam_checkpoint)
        self.sam_model = Sam2Model.from_pretrained(sam_checkpoint, device_map=self.device)
        
    # @util.timer_eval
    # @util.process_logger 
    def parse_detector_pred(self, text_prompt: str, results):
        # Parse Predictions into SAM Positive Boxes and Negative Points
        person_boxes = []
        negative_points = []
        text_prompt_list = list(map(lambda x: x.lower().strip(), text_prompt.split(".")))
        text_prompt_list  = list(filter(lambda x: len(x), text_prompt_list))

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = box.cpu().numpy()
            # Grounding DINO joins labels using the text prompt tokens
            _label = label.lower().strip()
            if len(list(filter(lambda x: x in _label, text_prompt_list))):
                person_boxes.append(box)
            else:
                # For non-person items overlapping the space, grab their center points
                # to use as "Do Not Include" markers in SAM
                center_x = int((box[0] + box[2]) / 2)
                center_y = int((box[1] + box[3]) / 2)
                negative_points.append([center_x, center_y])
                
        return person_boxes, negative_points
    
    # @util.timer_eval
    # @util.process_logger 
    def grounddino_sam_mask(self, image: util.PIL.Image.Image, text_prompt:str="a person.", text_prompt_negative:str="a gun. a knife. a bag.", threshold:float=0.20):
        inputs = self.detector_processor(images=image, text=text_prompt + " " + text_prompt_negative, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.detector_model(**inputs)

        # Post-process the detection to get raw bounding boxes
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.detector_processor.post_process_grounded_object_detection(
                            outputs, inputs.input_ids, 
                            threshold=threshold, 
                            # text_threshold=0.25, 
                            target_sizes=target_sizes
                        )[0]

        ###########################
        # Parse Predictions into SAM Positive Boxes and Negative Points
        person_boxes, negative_points = self.parse_detector_pred(text_prompt, results)
    
        if not person_boxes:
            return None

        # Assemble prompt vectors for SAM
        # SAM expects labels: 1 for foreground point/box, 0 for background/negative point
        input_boxes = [person_boxes]
        input_points = []
        input_labels = []
        # If we caught overlapping objects, append them as negative structural constraints
        if negative_points:
            # Filter negative points to match only if they fall inside or near a person's box
            input_points = [[negative_points] * len(person_boxes)] # 4 Levels
            input_labels = [[[0] * len(negative_points)] * len(person_boxes)] # 3 Levels
                    
        sam_inputs = self.sam_processor(images=image,
                                        input_boxes=input_boxes if input_boxes else None,
                                        input_points=input_points if input_points else None,
                                        input_labels=input_labels if input_labels else None,
                                        return_tensors="pt",
                                        ).to(self.device)
        ###########################
        with torch.no_grad():
            sam_outputs = self.sam_model(**sam_inputs)

        # final_masks shape is [N, 3, H, W]
        final_masks = self.sam_processor.post_process_masks(sam_outputs.pred_masks.cpu(),
                                                            sam_inputs["original_sizes"].cpu(),
                                                            mask_threshold=0.0,
                                                        )[0]

        # We pull the model's confidence scores [N, 3] to pick the highest scoring mask index.
        iou_scores = sam_outputs.iou_scores.cpu()[0]  # Shape: [N, 3]
        best_mask_indices = torch.argmax(iou_scores, dim=-1)  # Shape: [N]

        # Extract the best mask for each detected object
        # Slices the 3 choices down to the 1 best choice -> Resulting Shape: [N, H, W]
        N, _, H, W = final_masks.shape
        selected_masks = final_masks[torch.arange(N), best_mask_indices]

        # Convert from Boolean to Grayscale uint8 tensor (0 for background, 255 for person)
        grayscale_masks = selected_masks.to(dtype=torch.uint8).detach().numpy()
        grayscale_masks = (grayscale_masks > 0).astype(bool)
        # print(f"Final clean grayscale mask shape: {grayscale_masks.shape}")  # Output: [N, H, W]

        return grayscale_masks

    def predict(self, x:util.PIL.Image.Image | np.ndarray, **kwargs):
        if isinstance(x, np.ndarray):
            x = util.PIL.Image.fromarray(x)
        return self.grounddino_sam_mask(x, **kwargs)
    
class PerFrameObjectDetector():
    """
    PerFrameObjectDetector provides per-frame object detection and segmentation
    using configurable detection models (currently supports YOLO).
    """
    def __init__(self, detector_name: str = 'yolo', **kwargs) -> None:
        """
        Initialize the object detector with the specified detector name and parameters.

        Args:
            detector_name (str): Name of the detection tool/model (default: 'yolo').
            **kwargs: Additional configuration parameters for the detector.
        """
        self.detector_name = detector_name
        self.kwargs = kwargs

    def load_detector(self):
        """
        Load and initialize the object detection model based on configuration.

        This method sets up the segmentation tool, model name, and model parameters.
        Currently, only YOLO is supported. Raises an error for undefined models.

        Kwargs Args:
            detector_name (str): Name of the detection tool/model (default: 'yolo').
            detector_version (str): Version of the detection model to use (default: None).
            detector_version_kwargs (dict): Additional keyword arguments for the detection model (default: {}).
            seg_type (str): Type of segmentation to perform (default: 'instant').
        """        
        print('loading object detector...')
        print(self.kwargs)
        if self.detector_name is None:
            print("detector_name: ", self.detector_name)
            return

        self.seg_type = self.kwargs.get('seg_type', 'instant')
        self.detector_version = self.kwargs.get('detector_version', None)
        self.detector_version_kwargs = self.kwargs.get('detector_version_kwargs', {})
        self.seg_tool_dict = cfg.od_model_dict[self.detector_name]
        self.seg_model_name = self.seg_tool_dict[self.seg_type]['model'] if self.detector_version is None else self.detector_version
        self.seg_model_kwargs = self.seg_tool_dict[self.seg_type]['kwargs'] if self.detector_version_kwargs is None else self.detector_version_kwargs
        self.seg_model_kwargs.update(self.kwargs)
        device = self.seg_model_kwargs.get('device', 'cuda')

        print("model: ", self.seg_model_name)
        if self.detector_name in ['yolo']:
            from ultralytics import YOLO
            # YOLO-pretrained Instance segmentation model
            self.model = YOLO(f'{self.seg_model_name}.pt') # load a pretrained model
            
            if self.seg_model_kwargs.get('sahi_kwargs', None) is not None:
                print("sahi_kwargs: ", self.seg_model_kwargs)
                from sahi import AutoDetectionModel
                from sahi.predict import get_sliced_prediction as sahi_get_sliced_prediction
                self.sahi_get_sliced_prediction = sahi_get_sliced_prediction
                self.model_yolo_sahi = AutoDetectionModel.from_pretrained(
                    model_type="ultralytics",
                    model=self.model,
                    confidence_threshold=self.seg_model_kwargs.get('detection_thr', 0.25),
                    device=device  # or 'cuda:0'
                )

        elif self.detector_name == 'grounddino_sam':
            try:
                if self.detector_version is not None:
                    grounddino_modelname, sam_modelname = self.detector_version.split('__')
                else:
                    grounddino_modelname = self.seg_model_name[0]
                    sam_modelname = self.seg_model_name[1]
                self.model = GroundDinoSAMAN(grounddino_modelname=grounddino_modelname, 
                                sam_modelname=sam_modelname, 
                                device=device)
            except Exception as ex:
                print(f"Grounddino_sam detector model loading failed! {ex}")
                
        elif self.detector_name == 'rfdetr':
            self.model  = RFDETR(modelname=self.detector_version, device=device)

        else:
            raise "Undefined seg model. Please choose from ['yolo', 'rfdetr', 'grounddino_sam']."

    def get_detection(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Perform object detection and segmentation on a single frame.

        Args:
            frame (np.ndarray): Input image frame (BGR).

        Returns:
            tuple: (processed_frame, boolean_masks)
                processed_frame (np.ndarray): Possibly resized/restored frame.
                boolean_masks (np.ndarray or None): Boolean masks for detected objects.
        """
        if self.detector_name is None:
            return frame, None
        
        detection_thr = self.seg_model_kwargs.get('detection_thr', 0.25)
        classes = self.seg_model_kwargs.get('classes', [0])
        imgsz = self.seg_model_kwargs.get('imgsz', None)
        sahi_kwargs = self.seg_model_kwargs.get('sahi_kwargs', None)
        device = self.seg_model_kwargs.get('device', 'cuda')
        device = device if torch.cuda.is_available() else 'cpu'

        if self.detector_name in ['yolo']:
            # if imgsz is not None:
            #     _imgsz_org = frame.shape[:-1]
            #     frame = util.image_resize(frame, imgsz, ref="w") # preserves dim ratio

            _imgsz_org = frame.shape[:-1]
            is_imgsz = imgsz is not None and tuple(_imgsz_org) != tuple(imgsz)
            if is_imgsz:
                _imgsz_org = frame.shape[:-1]
                frame = util.image_resize(frame, imgsz, ref="w") # preserves dim ratio

            if sahi_kwargs is not None:
                
                def get_yolo_compatible_masks(results, classes=[0], target_shape=None):
                    """Converts SAHI sliced prediction masks into a PyTorch tensor compatible
                    with YOLO's `results[0].masks.data` format ([N, H, W]).
                    
                    Parameters:
                    - results: The object returned by `sahi_get_sliced_prediction(...)`
                    - target_shape: Tuple of (height, width) of the original image.
                                    If None, it tries to infer it from SAHI's internal metadata.

                    Returns:
                    - A np.array of shape [N, H, W] containing binary masks, or None if no
                    masks found.
                    """
                    # print("extracting mask from sahi...")
                    # Extract the object prediction list from SAHI
                    if not results:
                        return None
                    
                    mask_list = []
                    for pred in list(filter(lambda obj: obj.category.id in classes, results)):
                        if pred.mask is not None:
                            binary_mask = pred.mask.bool_mask
                            if isinstance(binary_mask, np.ndarray):
                                mask_list.append(binary_mask.astype(bool))

                    if not mask_list:
                        return None
                    # Stack into a single numpy array [N, H, W]
                    boolean_masks = np.stack(mask_list, axis=0)
                    return boolean_masks

                # tile-slicing based inferences
                results = self.sahi_get_sliced_prediction(frame.copy(),
                                    self.model_yolo_sahi,
                                    # slice_height=640,
                                    # slice_width=640,
                                    # overlap_height_ratio=0.2,
                                    # overlap_width_ratio=0.2,
                                    verbose=0,
                                    **sahi_kwargs
                                ).object_prediction_list
                
                boolean_masks = get_yolo_compatible_masks(results, classes=classes, target_shape=None)
                if is_imgsz and boolean_masks is not None:
                    boolean_masks = np.array([[cv2.resize(mask, _imgsz_org[::-1]).astype(bool) for mask in boolean_masks]])
                elif boolean_masks is not None:
                    boolean_masks = boolean_masks[np.newaxis,...]
            else:
                results = self.model.predict(frame, 
                                            conf=detection_thr, retina_masks=True, half=False,
                                            classes=classes, device=device, save=False, verbose=False
                                            )
                boolean_masks = None
                if results[0].masks: 
                    boolean_masks = results[0].masks.data.type(torch.uint8).detach().cpu().numpy()
                    if is_imgsz: 
                        boolean_masks = np.array([[cv2.resize(mask, _imgsz_org[::-1]).astype(bool) for mask in boolean_masks]])
                    else:
                        boolean_masks = np.array([[cv2.resize(mask, results[0].orig_shape[::-1]).astype(bool) for mask in boolean_masks]])
            if is_imgsz:
                frame = util.image_resize(frame, _imgsz_org[::-1], ref="w")
            
            return frame.astype("uint8"), boolean_masks

        elif self.detector_name == 'grounddino_sam':
            prompts_negative = self.seg_model_kwargs.get('prompt_negative', ["gun", "knife", "bag"])
            boolean_masks = self.model.predict(frame, 
                                               text_prompt=" ".join([f"a {cfg.od_classnames[c_idx]}." for c_idx in classes]), #"a person.", 
                                               text_prompt_negative=" ".join([f"a {prompt}." for prompt in prompts_negative]), #"a gun. a knife. a bag.", 
                                               threshold=0.35 if detection_thr is None else detection_thr
                                             )
            if boolean_masks is not None:
                return frame.astype("uint8"), boolean_masks[np.newaxis,...]
            else:
                return frame.astype("uint8"), boolean_masks
            
        elif self.detector_name == 'rfdetr':
            boolean_masks = self.model.predict(frame, 
                                    threshold=0.5 if detection_thr is None else detection_thr,  
                                    classes=[cfg.od_classnames[c_idx] for c_idx in classes]
                                    )
            if boolean_masks is not None:
                return frame.astype("uint8"), boolean_masks[np.newaxis,...]
        else:
            boolean_masks = [np.zeros(frame.shape, dtype=np.uint8)]
        return frame.astype("uint8"), boolean_masks
    
