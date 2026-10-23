# =============================================================================
# XAVAD: Unified Explainable Anonymized Video Anomaly Detection (VAD) System
# ZSAC: Zero-Shot Anomaly Categorization Generation
# =============================================================================
# This code defines a Python class for integrating the CLIP model into a XAVAD. 
# The class, CLIP4VAD_XAI, loads a specified CLIP model variant, sets up preprocessing for images, and prepares the model for feature extraction on video frames. 
# It supports different pooling methods for aggregating frame-level features into video-level representations, which are useful for detecting anomalies and generating explanations. 
# The code also loads prompt concepts for explainability, and prints key model parameters for transparency. 
# This setup is foundational for building systems that can detect and explain anomalies in video data while supporting privacy-preserving visual processing.

# Author: Mulugeta W. Asres
# Email: muleina2000@gmail.com
# Date: October 2025
#####################################################################

import numpy as np
import torch, cv2
from tqdm import tqdm
import clip as MULTI_MODAL_CLIP
from torch.utils.data import DataLoader

import utilities as util
from typing import List, Tuple, Dict, Union, Optional, Literal, Any

class CLIP4VAD_XAI():
    """ CLIP model for video anomaly detection and explainability.
    Args:
        1) clip_modelname: CLIP model name, options are "ViT-B/32","ViT-B/16","ViT-L/14", "ViT-L/14@336px"
        2) device: device to use for encoding, options are "cuda", "cpu"
        3) video_chunck_freq: number of frames to aggregate to a video snippet feature
        4) pooling_method: method to aggregate frame-level features to video-level feature, options
        are "mean", "attn" 
    """

    def __init__(self, clip_modelname: str = "ViT-B/16", pooling_method: str = "mean", prompt_concept_json_filepath: Union[str, Dict[str, str]] = 'xd-concept.json', prompt_feature_np_filepath: str = 'xd-prompt.npy', similarity_metric: str = 'cosine_softmax', device: str = "cuda", **kwargs):
        """ Initialize CLIP model for video anomaly detection and explainability.
        Args: 
            clip_modelname: CLIP model name, options are "ViT-B/32","ViT-B/16","ViT-L/14", "ViT-L/14@336px"
            device: device to use for encoding, options are "cuda", "cpu"
            video_chunck_freq: number of frames to aggregate to a video snippet feature
            pooling_method: method to aggregate frame-level features to video-level feature, options are "mean", "attn"
        """
        print("CLIP4VAD_XAI init...")
        self.device = device if torch.cuda.is_available() else "cpu"
        self.clip_modelname = clip_modelname
        self.pooling_method = pooling_method
        self.similarity_metric = similarity_metric
        self.prompt_concept_json_filepath = prompt_concept_json_filepath
        self.prompt_feature_np_filepath = prompt_feature_np_filepath 
        self.isreturn_max_indices = kwargs.get("isreturn_max_indices", False)
        self.clip_model, self.clip_image_preprocess = MULTI_MODAL_CLIP.load(clip_modelname)
        self.clip_model = self.clip_model.to(self.device).eval()
        self.clip_input_resolution = self.clip_model.visual.input_resolution
        self.clip_ksize = self.clip_model.visual.conv1.kernel_size
        self.context_length = self.clip_model.context_length
        self.vocab_size = self.clip_model.vocab_size
        
        print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in self.clip_model.parameters()])}")
        print("Input resolution:", self.clip_input_resolution)
        print("Kernel size:", self.clip_ksize)
        print("Context length:", self.context_length)
        print("Vocab size:", self.vocab_size)
        print("Pooling method:", self.pooling_method)
        print("CLIP image preprocess function: ", self.clip_image_preprocess)
        self.anml_class_names, self.anml_class_prompt_clip_features_norm = self.load_clip_anml_prompt_features()

    def load_clip_anml_prompt_features(self):
        """ Load CLIP prompt features for video anomaly detection.
        Returns:
            anml_class_names: list of anomaly class names
            anml_class_prompt_clip_features_norm: tensor of anomaly class prompt features
        """
        if isinstance(self.prompt_concept_json_filepath, dict):
            print("Generating CLIP text embedding....")
            anml_type_prompt_concept_dict = self.prompt_concept_json_filepath
            anml_class_names = list(anml_type_prompt_concept_dict.keys())
            anml_class_prompt_clip_features_norm = self.clip_model.encode_text(MULTI_MODAL_CLIP.tokenize(["" + desc for name, desc in anml_type_prompt_concept_dict.items()]).to(self.device)).float()
            anml_class_prompt_clip_features_norm /= anml_class_prompt_clip_features_norm.norm(dim=-1, keepdim=True)
            anml_class_prompt_clip_features_norm = anml_class_prompt_clip_features_norm.T
        else:
            print("Loading Pre-computed Normalized CLIP text embedding....")
            anml_type_prompt_concept_dict = util.load_json(None, filepath=self.prompt_concept_json_filepath)
            anml_class_names = list(anml_type_prompt_concept_dict.keys())
            anml_class_prompt_clip_features_norm = util.load_npdata(None, filepath=self.prompt_feature_np_filepath)
            anml_class_prompt_clip_features_norm = torch.tensor(anml_class_prompt_clip_features_norm, dtype=torch.float32).to(self.device).T
        
        self.anml_type_prompt_concept_dict = anml_type_prompt_concept_dict
        print(f"CLIP Prompt Anomaly Categories: {list(self.anml_type_prompt_concept_dict.keys())}")
        print(f"anml_class_prompt_clip_features_norm: {anml_class_prompt_clip_features_norm.shape}")
        return anml_class_names, anml_class_prompt_clip_features_norm
    
    # @util.timer_eval
    # @util.process_logger
    def clip_video_preprocessor(self, video_frames, video_metadata: dict = {}):
        """ Encode video frames using CLIP model and aggregate the frame-level features to video-level feature.
        Args:
            video_frames: list of video frames (numpy array)
            video_metadata: dictionary of video metadata
        Returns:
            images: list of preprocessed video frames (tensor)
        """

        print("video frames preprocessing...")
        video_format = video_metadata.get("video_format", "RGB")
        images = []
        for frame in tqdm(video_frames):
            image = util.PIL.Image.fromarray(cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2RGB) if video_format == "BGR" else frame.astype(np.uint8))
            if video_format == "RGBA":
                image = image.convert("RGB")
            images.append(self.clip_image_preprocess(image))
        return images

    # @util.timer_eval
    # @util.process_logger
    def clip_video_encoder(self, video_frames, video_metadata: dict = {}, batch_size: int = 8, video_chunck_freq: int = 16, **kwargs):
        """ Encode video frames using CLIP model and aggregate the frame-level features to video-level feature.
        Args:
            video_frames: list of video frames (numpy array)
            video_metadata: dictionary of video metadata
            batch_size: batch size for encoding
            device: device to use for encoding
            video_chunck_freq: number of frames to aggregate to a video snippet feature
            pooling_method: method to aggregate frame-level features to video-level feature, options are "mean", "attn"
        Returns:
            video_clip_features: video-level feature (tensor)
        """
        print(f"Encoding video frames using CLIP model: {self.clip_modelname}, batch_size: {batch_size}, device: {self.device}, video_chunck_freq: {video_chunck_freq}, pooling_method: {self.pooling_method}")

        # use the class method for preprocessing
        images = self.clip_video_preprocessor(video_frames, video_metadata)

        print("CLIP encoding video frames...")
        images_clip_features = []
        for images_batch in tqdm(DataLoader(images, batch_size=batch_size, shuffle=False, drop_last=False)):
            # gc.collect()
            # torch.cuda.empty_cache()
            with torch.no_grad():
                image_features = self.clip_model.encode_image(torch.tensor(np.stack(images_batch)).to(self.device)).float()
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                images_clip_features.append(image_features)

        images_clip_features = torch.cat(images_clip_features, dim=0)

        vide_clipped_length = ((images_clip_features.shape[0]) // video_chunck_freq) * video_chunck_freq  # based on i3d extractor setting
        images_clip_feature_chunks = torch.split(images_clip_features[:vide_clipped_length], video_chunck_freq, dim=0)
        
        def mean_pool(x):
            return x.mean(dim=0, keepdim=True)
        def median_pool(x):
            return x.median(dim=0, keepdim=True)[0]
        def max_pool(x):
            return x.unsqueeze(0)

        if self.pooling_method == "mean":
            video_clip_features = util.multiprocessor_speedup(mean_pool, input_data=images_clip_feature_chunks, num_workers=video_chunck_freq)
        elif self.pooling_method == "median":
            video_clip_features = util.multiprocessor_speedup(median_pool, input_data=images_clip_feature_chunks, num_workers=video_chunck_freq)
        else:
            video_clip_features = util.multiprocessor_speedup(max_pool, input_data=images_clip_feature_chunks, num_workers=video_chunck_freq)
        
        video_clip_features = torch.cat(video_clip_features, dim=0)          

        video_clip_features /= video_clip_features.norm(dim=-1, keepdim=True) # added jan 14, 2025
        
        return video_clip_features
    
    # @util.timer_eval
    # @util.process_logger
    def get_clip_class_similarity_scores_labels(self, video_clip_features_norm, top_k: int = 3):
        """ Calculate similarity scores between video clip features and class prompt features, and get top-k class labels.
        Args:
            video_clip_features_norm: video clip features (tensor)
            top_k: number of top class labels to return
        Returns:
            top_class_top_probs: top-k class probabilities (tensor)
            top_class_top_labels: top-k class labels (tensor)
        """
        similarity_max_indices = None
        if video_clip_features_norm.ndim > 2:
            vid_emb_with_crop_shape = video_clip_features_norm.shape
            video_clip_features_norm = video_clip_features_norm.view((-1, vid_emb_with_crop_shape[-1]))
            similarity = video_clip_features_norm @ self.anml_class_prompt_clip_features_norm
            similarity = similarity.view((vid_emb_with_crop_shape[0], vid_emb_with_crop_shape[1], -1))
            similarity_scores, similarity_max_indices = similarity.max(dim=1)
            similarity_max_indices = similarity_max_indices.detach().cpu()
        else:
            similarity_scores = video_clip_features_norm @ self.anml_class_prompt_clip_features_norm
            
        class_probs = (100.0 * similarity_scores).softmax(dim=-1) # to match classification final layer
        top_anml_class_top_probs, top_anml_class_top_labels = class_probs.detach().cpu().topk(top_k, dim=-1)
        similarity_scores = similarity_scores.detach().cpu()
        return top_anml_class_top_probs, top_anml_class_top_labels, (similarity_scores, similarity_max_indices)
    
    def get_class_names_from_labels(self, class_idx_labels, class_probs, video_chunck_freq: int = 16):
        """ Get class names from class index labels and class probabilities.
        Args:
            class_idx_labels: class index labels (tensor)
            class_probs: class probabilities (tensor)
            video_chunck_freq: number of frames to to extend to a video snippet feature
        Returns:
            class_names_preds: list of dictionaries of class names and probabilities
        """
        print(f"Getting class names from labels, video_chunck_freq: {video_chunck_freq}")
        class_names_preds = []
        for class_idx_label_i, class_prob_i in zip(class_idx_labels, class_probs):
            class_names_preds.append({self.anml_class_names[class_label_idx]:np.round(float(class_prob_i[j]), 5) for j, class_label_idx in enumerate(class_idx_label_i)})

        class_names_preds = [class_names_pred for class_names_pred in class_names_preds for _ in range(video_chunck_freq)] # repeat for each video_chunck_freq to macth original frame length 
        return class_names_preds

    @util.timer_eval
    @util.process_logger
    def inference(self, video_frames, video_metadata, batch_size: int = 8, video_chunck_freq: int = 16, top_k: int = 3, video_clip_features_norm: Optional[np.ndarray] = None, **kwargs):
        """ Inference video frames using CLIP model for video anomaly detection and explainability.
        Args:
            video_frames: list of video frames (numpy array)
            video_metadata: dictionary of video metadata
            batch_size: batch size for encoding
            video_chunck_freq: number of frames to aggregate to a video snippet feature
            top_k: number of top class labels to return
        Returns:
            top_class_top_probs: top-k class probabilities (tensor)
            top_class_top_labels: top-k class labels (tensor)
            class_names_preds: list of dictionaries of class names and probabilities
        """
        # clip vid feature extraction
        if video_clip_features_norm is None:
            video_clip_features_norm = self.clip_video_encoder(video_frames,
                                            video_metadata=video_metadata,
                                            video_chunck_freq=video_chunck_freq,
                                            batch_size=batch_size, 
                                                **kwargs)
        
        # clip vid class similarity and top-k class names prediction
        top_anml_class_probs, top_anml_class_top_labels, similarity_scores = self.get_clip_class_similarity_scores_labels(video_clip_features_norm, top_k=top_k)
        top_anml_class_probs, top_anml_class_top_labels = top_anml_class_probs.detach().numpy(), top_anml_class_top_labels.detach().numpy()
        if self.isreturn_max_indices and similarity_scores[1] is not None:
            similarity_scores = [s.detach().numpy() for s in similarity_scores]
        else:
            similarity_scores = similarity_scores[0].detach().numpy()
            
        ad_class_names_preds = self.get_class_names_from_labels(top_anml_class_top_labels, top_anml_class_probs, video_chunck_freq=video_chunck_freq)
        return top_anml_class_probs, top_anml_class_top_labels, ad_class_names_preds, similarity_scores, video_clip_features_norm
    
class ADXAIDetector():
    
    def __init__(self, xai_clip_modelname: str = "ViT-L/14", xai_anml_cat_list: Optional[List[str]] = None, video_chunck_freq: int = 16, pooling_method: str = "max", device: Literal["cuda", "cpu"] = "cuda", **kwargs):
        self.xai_clip_modelname = xai_clip_modelname
        self.xai_anml_cat_list = xai_anml_cat_list
        self.video_chunck_freq = video_chunck_freq
        self.pooling_method = pooling_method
        self.isreturn_max_indices = kwargs.get("isreturn_max_indices", False)
        
        if self.xai_anml_cat_list is None or len(self.xai_anml_cat_list) == 0:
            self.prompt_concept_json_filepath = kwargs.get("prompt_concept_json_filepath", r"/home/mulugetawa/AI4CITIZENS/PACA/src/model/clip/llmv2_custom-concept.json")
            self.prompt_feature_np_filepath = kwargs.get("prompt_feature_np_filepath", rf"/home/mulugetawa/AI4CITIZENS/PACA/src/model/clip/anml_class_prompt_llm_clip_features_norm_llmv2_custom_{self.xai_clip_modelname.replace('-','').replace('/','')}.npy")
        else:
            print(f"xai_anml_cat_list: {self.xai_anml_cat_list}")
            if isinstance(self.xai_anml_cat_list, dict):
                self.prompt_concept_json_filepath = {k:f"{v['prompt']}" for k, v in self.xai_anml_cat_list.items()}
            elif isinstance(self.xai_anml_cat_list, list):
                self.prompt_concept_json_filepath = {k:"a photo of " +' '.join(k.split('_')) for k in self.xai_anml_cat_list}
                            
            self.prompt_feature_np_filepath = None
          
        clip4vadObj = CLIP4VAD_XAI(clip_modelname=self.xai_clip_modelname,
                                pooling_method=self.pooling_method, 
                                prompt_concept_json_filepath=self.prompt_concept_json_filepath, 
                                prompt_feature_np_filepath=self.prompt_feature_np_filepath, 
                                isreturn_max_indices=self.isreturn_max_indices,
                                device=device
                                )
        self.clip4vadObj = clip4vadObj
        self.anml_text_prompt_embedding = self.clip4vadObj.anml_class_prompt_clip_features_norm.type(torch.float16).to(device)
    
    # @util.clean_cuda
    # @util.timer_eval
    # @util.process_logger   
    def inference(self, video_frames, top_k=1, **kwargs):
        xai_ad_type_isplot = kwargs.get("xai_ad_type_isplot", True)
        xai_ad_map_isplot = kwargs.get("xai_ad_map_isplot", False)
        video_metadata = kwargs.pop("video_metadata", {})
        video_chunck_freq = kwargs.pop("video_chunck_freq", len(video_frames))

        if xai_ad_type_isplot or xai_ad_map_isplot:
            # clip vid class similarity and top-k class names prediction
            _, top_anml_class_top_labels, ad_class_names_preds, similarity_scores, _ = self.clip4vadObj.inference(video_frames[-video_chunck_freq:], video_metadata, 
                                                                                              batch_size=video_chunck_freq, 
                                                                                              video_chunck_freq=video_chunck_freq,
                                                                                                top_k=top_k, **kwargs)
            # print(ad_class_names_preds)
            if self.isreturn_max_indices:
                return ad_class_names_preds[0], top_anml_class_top_labels[0], similarity_scores
            return ad_class_names_preds[0], top_anml_class_top_labels[0]
        else:
            return None, None
    

