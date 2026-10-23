# =============================================================================
# XAVAD: Unified Explainable Anonymized Video Anomaly Detection (VAD) System
# =============================================================================
# This code defines a Python class for integrating the Grad-ECLIP model into a XAVAD. 
# The class, GRAD4VAD_XAI, loads a specified CLIP model variant, sets up preprocessing for images, and prepares video frame, and extractes anomlay heatmap

# Author: Mulugeta W. Asres
# Email: muleina2000@gmail.com
# Date: October 2025
#####################################################################
import argparse
import sys, gc, os, io
import numpy as np, pandas as pd
import torch, gc, cv2
import matplotlib.pyplot as plt
import moviepy.editor as mpy
import clip as MULTI_MODAL_CLIP
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
from typing import List, Tuple, Union, Optional, Literal, Dict
import supervision as sv
import scipy.ndimage as ndi
from ultralytics.engine.results import Results

import utilities as util
from xai_vad_zsac_interface import ADXAIDetector


def visualize_grad_cam_per_image(img_pil, filtered_grad_heatmaps, fig_titles=None, fig_scores=None, fig_rows=1, **kwargs):
    overlap_alpha = kwargs.get("overlap_alpha", 0.5)
    vis_resize_wh = kwargs.get("vis_resize_wh", None)
    fix_label_rec = kwargs.get("fix_label_rec", False)
    ad_thr = kwargs.get("ad_thr", 0.5)
    
    if vis_resize_wh is None:
        w, h = img_pil.size
    else:
        w, h = vis_resize_wh
    
    resize = lambda x: cv2.resize(x, (w, h), interpolation=cv2.INTER_LINEAR)
    image_np = np.asarray(img_pil) if vis_resize_wh is None else resize(np.asarray(img_pil))

    # w, h = img_pil.size
    # resize = lambda x: cv2.resize(x, (w, h), interpolation=cv2.INTER_LINEAR)
    R = 10
    TITLE_CENTER = (R, 1*R) # x, y
    REC_TITLE_CORRD = ((TITLE_CENTER[0]-R, TITLE_CENTER[1]-1*R), (TITLE_CENTER[0]+45*R, TITLE_CENTER[1]+R))
    SCORE_CENTER = (R, REC_TITLE_CORRD[1][1]) # x, y
    REC_SCORE_CORRD = ((SCORE_CENTER[0]-R, SCORE_CENTER[1]-1*R), (SCORE_CENTER[0]+45*R, SCORE_CENTER[1]+R))
    TEXT_FONT =  cv2.FONT_HERSHEY_TRIPLEX
    TEXT_SCALE = 0.7 # >0
    _TEXT_SCALE = 1 # >0
    TEXT_THICKNESS = 1
    frame_list = []
    x, y = TITLE_CENTER
    
    PERC_SCORE_MAX_LEN = 6
    vad_score_txt = "DETECTION:  "
    _text_size, _ = cv2.getTextSize(vad_score_txt, TEXT_FONT, TEXT_SCALE, TEXT_THICKNESS)
    _text_w, _ = _text_size
    vad_score_txt = "{:<{}}".format(vad_score_txt, 28)
    VAD_SCORE_MIN_LEN = REC_SCORE_CORRD[0][0] + _text_w
    VAD_SCORE_MAX_LEN = REC_SCORE_CORRD[1][0] - _text_w - PERC_SCORE_MAX_LEN - VAD_SCORE_MIN_LEN
    
    for i, c_ret in enumerate(filtered_grad_heatmaps):
        if c_ret is None:
            c_ret_overlayed = image_np
        else:
            c_ret_overlayed = np.clip(image_np * (1 - overlap_alpha) + c_ret * overlap_alpha, 0, 255).astype("uint8")
            filter_map = c_ret == 0
            c_ret_overlayed[filter_map] = image_np[filter_map]
            
        c_ret_overlayed = cv2.cvtColor(c_ret_overlayed, cv2.COLOR_RGB2BGR)
        
        if fig_scores is not None:
            _vad_score_txt = vad_score_txt + "{:>{}}%".format(f"{fig_scores[i]*100:0.1f}", PERC_SCORE_MAX_LEN)
            text_size, _ = cv2.getTextSize(_vad_score_txt, TEXT_FONT, TEXT_SCALE, TEXT_THICKNESS)
            text_w, text_h = text_size
            _ = cv2.rectangle(c_ret_overlayed, REC_TITLE_CORRD[0], (REC_TITLE_CORRD[1][0] if fix_label_rec else x + text_w + R, y + text_h + 1*R), (209, 222, 221), -1, lineType=cv2.LINE_8, shift=0)
            _ = cv2.putText(c_ret_overlayed, _vad_score_txt, (x, y + text_h + _TEXT_SCALE - 1), TEXT_FONT, TEXT_SCALE, (255, 0, 0), TEXT_THICKNESS, cv2.LINE_AA)
            _ = cv2.rectangle(c_ret_overlayed, (VAD_SCORE_MIN_LEN, y + int(0.0*R)), (VAD_SCORE_MIN_LEN + int(fig_scores[i]*VAD_SCORE_MAX_LEN), y + int(2.0*R)), (0, 0, 255) if fig_scores[i] > ad_thr else (0, 165, 255), -1, lineType=cv2.LINE_8, shift=0)
        else:
            text_w, text_h = 0, 0
            
        if fig_titles is not None:
            _text_size, _ = cv2.getTextSize(fig_titles[i], TEXT_FONT, TEXT_SCALE, TEXT_THICKNESS)
            _text_w, _text_h = _text_size
            _ = cv2.rectangle(c_ret_overlayed, (REC_SCORE_CORRD[0][0], y + text_h + 1*R), (REC_SCORE_CORRD[1][0] if fix_label_rec else x + _text_w + R, y + text_h + _text_h + 2*R), (255, 255, 255), -1, lineType=cv2.LINE_8, shift=0)
            _ = cv2.putText(c_ret_overlayed, fig_titles[i], (x, y + text_h + 3*R + _TEXT_SCALE - 1), TEXT_FONT, TEXT_SCALE, (255, 0, 0), TEXT_THICKNESS, cv2.LINE_AA)
        
        frame_list.append(cv2.cvtColor(c_ret_overlayed, cv2.COLOR_BGR2RGB))
  
    fig = util.plot_image_grid(frame_list, fig_rows, len(filtered_grad_heatmaps)//fig_rows, return_fig=True, border=0 if len(filtered_grad_heatmaps) <= 1 else 1, color_format="rgb")
    return fig


def show_image(fig_pil, isplot=True):
    if isplot:
        plt.imshow(fig_pil)
        plt.grid(False) 
        plt.axis('off')
        plt.show()


class GRAD4VAD_XAI():
    """  XAI MAP generating class based in GRAD-ECLIP for video anomaly detection explainability.
    Args:
        1) clip_modelname: CLIP model name, options are "ViT-B/32","ViT-B/16","ViT-L/14", "ViT-L/14@336px"
        2) device: device to use for encoding, options are "cuda", "cpu"
    """
    @util.process_logger
    def __init__(self, clip_modelname="ViT-B/16", device="cuda", **kwargs):
        """ Initialize CLIP model for video anomaly detection and explainability.
        Args: 
            clip_modelname: CLIP model name, options are "ViT-B/32","ViT-B/16","ViT-L/14", "ViT-L/14@336px"
            device: device to use for encoding, options are "cuda", "cpu"
            video_chunck_freq: number of frames to aggregate to a video snippet feature
            pooling_method: method to aggregate frame-level features to video-level feature, options are "mean", "attn"
        """
        self.device = device if torch.cuda.is_available() else "cpu"

        self.clip_modelname = clip_modelname
        self.clip_model, self.clip_image_preprocess = kwargs.get("clip_model_obj", MULTI_MODAL_CLIP.load(self.clip_modelname, device=self.device))
        self.clip_model = self.clip_model.to(self.device).eval()
        self.clip_input_resolution = self.clip_model.visual.input_resolution
        self.clip_ksize = self.clip_model.visual.conv1.kernel_size
        self.context_length = self.clip_model.context_length
        self.vocab_size = self.clip_model.vocab_size

        self.pre_transform = Compose([
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
                
        # print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in self.clip_model.parameters()])}")
        # print("Input resolution:", self.clip_input_resolution)
        # print("Clip kernel size:", self.clip_ksize)
        # print("Context length:", self.context_length)
        # print("Vocab size:", self.vocab_size)

        # Image Preprocessing
        # print("CLIP image preprocess function (not used): ", self.clip_image_preprocess)
    
    def clip_encode_dense(self, x, n):
        def attention_layer(q, k, v, num_heads=1):
            "Compute 'Scaled Dot Product Attention'"
            tgt_len, bsz, embed_dim = q.shape
            head_dim = embed_dim // num_heads
            scaling = float(head_dim) ** -0.5
            q = q * scaling
            q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
            k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
            v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
            attn_output_weights = torch.bmm(q, k.transpose(1, 2))
            attn_output_weights = F.softmax(attn_output_weights, dim=-1)
            attn_output_heads = torch.bmm(attn_output_weights, v)
            assert list(attn_output_heads.size()) == [bsz * num_heads, tgt_len, head_dim]
            attn_output = attn_output_heads.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, -1)
            attn_output_weights = attn_output_weights.sum(dim=1) / num_heads
            return attn_output, attn_output_weights
    
        vision_width = self.clip_model.visual.transformer.width
        vision_heads = vision_width // 64
        # print("[vision_width and vision_heads]:", vision_width, vision_heads)
        
        # modified from CLIP
        x = x.half()
        x = self.clip_model.visual.conv1(x)  
        feah, feaw = x.shape[-2:]

        x = x.reshape(x.shape[0], x.shape[1], -1) 
        x = x.permute(0, 2, 1) 
        class_embedding = self.clip_model.visual.class_embedding.to(x.dtype)
        x = torch.cat([class_embedding + torch.zeros(x.shape[0], 1, x.shape[-1]).to(x), x], dim=1)

        ## scale position embedding as the image w-h ratio
        pos_embedding = self.clip_model.visual.positional_embedding.to(x.dtype)
        tok_pos, img_pos = pos_embedding[:1, :], pos_embedding[1:, :]
        pos_h = self.clip_input_resolution // self.clip_ksize[0]
        pos_w = self.clip_input_resolution // self.clip_ksize[1]
        # assert img_pos.size(0) == (pos_h * pos_w), f"the size of pos_embedding ({img_pos.size(0)}) does not match resolution shape pos_h ({pos_h}) * pos_w ({pos_w})"
        img_pos = img_pos.reshape(1, pos_h, pos_w, img_pos.shape[1]).permute(0, 3, 1, 2)
        # print("[POS shape]:", img_pos.shape, (feah, feaw))
        img_pos = torch.nn.functional.interpolate(img_pos, size=(feah, feaw), mode='bicubic', align_corners=False)
        img_pos = img_pos.reshape(1, img_pos.shape[1], -1).permute(0, 2, 1)
        pos_embedding = torch.cat((tok_pos[None, ...], img_pos), dim=1)
        x = x + pos_embedding
        x = self.clip_model.visual.ln_pre(x)
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = torch.nn.Sequential(*self.clip_model.visual.transformer.resblocks[:-n])(x)

        attns = []
        atten_outs = []
        vs = []
        qs = []
        ks = []
        for TR in self.clip_model.visual.transformer.resblocks[-n:]:
            x_in = x
            x = TR.ln_1(x_in)
            linear = torch._C._nn.linear    
            q, k, v = linear(x, TR.attn.in_proj_weight, TR.attn.in_proj_bias).chunk(3, dim=-1)
            attn_output, attn = attention_layer(q, k, v, 1)  # vision_heads=1
            attns.append(attn)
            atten_outs.append(attn_output)
            vs.append(v)
            qs.append(q)
            ks.append(k)
            
            x_after_attn = linear(attn_output, TR.attn.out_proj.weight, TR.attn.out_proj.bias)       
            x = x_after_attn + x_in
            x = x + TR.mlp(TR.ln_2(x))

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.visual.ln_post(x)
        x = x @ self.clip_model.visual.proj
        return x, x_in, vs, qs, ks, attns, atten_outs, (feah, feaw)

    def grad_eclip(self, c, qs, ks, vs, attn_outputs, map_size, c_idx=0):
        # gradient on last attention output
        
        def sim_qk(q, k):
            # generate ODAM map
            q_cls = F.normalize(q[:1,0,:], dim=-1) 
            k_patch = F.normalize(k[1:,0,:], dim=-1)
            cosine_qk = (q_cls * k_patch).sum(-1) 
            cosine_qk_max = cosine_qk.max(dim=-1, keepdim=True)[0]
            cosine_qk_min = cosine_qk.min(dim=-1, keepdim=True)[0]
            cosine_qk = (cosine_qk-cosine_qk_min) / (cosine_qk_max-cosine_qk_min)
            return cosine_qk
        
        tmp_maps = []

        for q, k, v, attn_output in zip(qs, ks, vs, attn_outputs):
            grad = torch.autograd.grad(
                c,
                attn_output,
                grad_outputs=None,
                retain_graph=True)[0]
            grad_cls = grad[:1,c_idx,:]
            v_patch = v[1:,c_idx,:]
            cosine_qk = sim_qk(q, k).reshape(-1)
            tmp_maps.append((grad_cls * v_patch * cosine_qk[:,None]).sum(-1)) 
        emap = F.relu_(torch.stack(tmp_maps, dim=0)).sum(0)
        return emap.reshape(*map_size)

    def self_attn(self, attns, map_size):
        attn_patch = attns[-1][0,:1,1:].reshape(*map_size)
        print("[attn of cls token on lastv]:", attn_patch.shape)
        return attn_patch

    def grad_cam(self, c, feat, map_size):
        ## GRAD-CAM: use the feature outputs of the final attention layer
        grad = torch.autograd.grad(
            c,
            feat,
            retain_graph=True)[0] 
        grad_weight = grad.mean(0, keepdim=True)
        grad_cam = F.relu_((grad_weight * feat).sum(-1))
        grad_cam = grad_cam[1:].reshape(*map_size)
        return  grad_cam   

    def prepare_grad_score_profile(self, img_pil, text_embedding):
        # text_embedding is transposed
        def imgprocess(img, patch_size=[16, 16], scale_factor=1):
            w, h = img.size
            ph, pw = patch_size
            nw = int(w * scale_factor / pw + 0.5) * pw
            nh = int(h * scale_factor / ph + 0.5) * ph

            ResizeOp = Resize((nh, nw), interpolation=InterpolationMode.BICUBIC)
            img = ResizeOp(img).convert("RGB")
            return self.pre_transform(img)
        
        img_preprocessed_k = imgprocess(img_pil, patch_size=self.clip_ksize).to(self.device).unsqueeze(0)
        outputs, last_feat, vs, qs, ks, attns, atten_outs, map_size = self.clip_encode_dense(img_preprocessed_k, n=1)
        img_embedding = F.normalize(outputs[:,0], dim=-1)
        similarity_score_cosine = (img_embedding @ text_embedding)[0].cpu()
        grad_score_profile = {"similarity_score_cosine":similarity_score_cosine, 
                            "last_feat":last_feat, 
                            "vs":vs, "qs":qs, "ks":ks, 
                            "attns":attns, "atten_outs":atten_outs, 
                            "map_size":map_size}
        return grad_score_profile
    
    # @util.timer_eval
    # @util.process_logger
    def grad_heatmap_clip_score_per_image(self, img_pil, text_embedding, grad_mode="grad_eclip"):
        grad_score_profile = self.prepare_grad_score_profile(img_pil, text_embedding)
        
        grad_heatmaps = []
        if grad_mode == "grad_cam":
            for i, c in enumerate(grad_score_profile["similarity_score_cosine"]):
                grad_heatmaps.append(self.grad_cam(c, grad_score_profile["last_feat"], grad_score_profile["map_size"]))
        elif grad_mode == "grad_eclip":
            for i, c in enumerate(grad_score_profile["similarity_score_cosine"]):
                grad_heatmaps.append(self.grad_eclip(c, grad_score_profile["qs"], grad_score_profile["ks"], grad_score_profile["vs"], 
                                                    grad_score_profile["atten_outs"], 
                                                    grad_score_profile["map_size"], c_idx=i)
                                        )
            
        else:
            raise "undefined mode={} for grad_heatmap_clip_score_per_image. please select from [grad_eclip, grad_cam]."

        w, h = img_pil.size
        resize = T.Resize((h, w))
        grad_heatmaps = [resize(grad_heatmap.unsqueeze(0)).detach().cpu().numpy()[0] for grad_heatmap in grad_heatmaps]
        return grad_heatmaps
    
    # @util.timer_eval
    # @util.process_logger
    def prepare_final_heatmap(self, img_pil, grad_heatmaps, verbose=0, **kwargs):
        map_score_thr_dict = kwargs.get("map_score_thr_dict", {})
        map_score_min = map_score_thr_dict.get("map_score_min", 0.03)
        map_score_max = map_score_thr_dict.get("map_score_max", 0.50)
        map_norm_score_min = map_score_thr_dict.get("map_norm_score_min", 0.10)
        if map_norm_score_min is not None:
            map_norm_score_min = 255*map_norm_score_min
        
        def postprocess_heatmap(map_score, resize=None, map_norm_score_min=0):
            map_score = (map_score*255).astype(np.uint8)
            map_score = map_score if resize is None else resize(map_score)
            color = cv2.applyColorMap(map_score, cv2.COLORMAP_JET) # cv2 to plt
            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            if map_norm_score_min is not None:
                color[map_score < map_norm_score_min] = 0
            return color
        
        vis_resize_wh = kwargs.get("vis_resize_wh", None)
        # w, h = img_pil.size
        if vis_resize_wh is None:
            w, h = img_pil.size
        else:
            w, h = vis_resize_wh
        resize = lambda x: cv2.resize(x, (w, h), interpolation=cv2.INTER_LINEAR)
        # image_np = np.asarray(img_pil) if vis_resize_wh is None else resize(np.asarray(img_pil))
        filtered_grad_heatmaps = []
        # null_gradmap = np.zeros(vis_resize_wh, dtype=np.uint8)
        null_gradmap = None
        for grad_heatmap in grad_heatmaps:
            if grad_heatmap is None:
                c_ret = null_gradmap
            else:    
                tmp = grad_heatmap.copy()
                tmp_min, tmp_max = np.min(tmp), np.max(tmp)
                if verbose: print(f"tmp_min: {tmp_min}, tmp_max: {tmp_max}")
                
                if map_score_min is not None:
                    if tmp_max < map_score_min:
                        c_ret = null_gradmap
                    else:
                        tmp[tmp < map_score_min] = 0
                        tmp_min = 0
                        # cut high score to capture all valid score above map_score_max
                        if map_score_max is not None: 
                            tmp[tmp > map_score_max] = map_score_max
                        tmp -= tmp_min
                        tmp /= np.max(tmp)       
                        c_ret = postprocess_heatmap(tmp, resize=resize, map_norm_score_min=map_norm_score_min)
                else:
                    # cut high score to capture all valid score above map_score_max
                    if map_score_max is not None: 
                        tmp[tmp > map_score_max] = map_score_max
                        tmp_min = 0
                    tmp -= tmp_min
                    tmp /= np.max(tmp)       
                    c_ret = postprocess_heatmap(tmp, resize=resize, map_norm_score_min=map_norm_score_min)
                    
            filtered_grad_heatmaps.append(c_ret)
        return filtered_grad_heatmaps
                    
    def inference(self, img_pil, anml_text_prompt_embedding, grad_mode: str ="grad_eclip", **kwargs):
        # anml_text_prompt_embedding = anml_text_prompt_embedding[:, ad_class_names_id_topk]
        grad_heatmaps = self.grad_heatmap_clip_score_per_image(img_pil, anml_text_prompt_embedding, grad_mode=grad_mode)
        filtered_grad_heatmaps = self.prepare_final_heatmap(img_pil, grad_heatmaps, **kwargs)
        return grad_heatmaps, filtered_grad_heatmaps
        
        
class ADXAIMAPDetector():
    
    def __init__(self, objVADXAIModel, device: Literal["cuda", "cpu"] = "cuda", **kwargs):
        
        self.visualize_grad_cam_per_image = visualize_grad_cam_per_image
        self.anml_text_prompt_embedding = objVADXAIModel.clip4vadObj.anml_class_prompt_clip_features_norm.type(torch.float16).to(device)
        self.anml_class_names = objVADXAIModel.clip4vadObj.anml_class_names
        self.map_score_thr_dict_default = dict(map_score_min=0.03, map_score_max=None, map_norm_score_min=0.10)

        print(f"MAP anml_text_prompt_embedding: {self.anml_text_prompt_embedding.shape}")
        
        self.gradmap4vadObj = GRAD4VAD_XAI(clip_modelname=objVADXAIModel.xai_clip_modelname,
                                        clip_model_obj=(objVADXAIModel.clip4vadObj.clip_model, objVADXAIModel.clip4vadObj.clip_image_preprocess),
                                        device=device
                                        )
    
    # @util.timer_eval
    # @util.process_logger
    def inference(self, image_pil, ad_class_names_id_topk: int, **kwargs):
        isget_map = kwargs.get("isget_map", False)
        map_score_thr_dict = kwargs.pop("map_score_thr_dict", self.map_score_thr_dict_default)
        
        if isinstance(image_pil, np.ndarray):
            image_pil = util.PIL.Image.fromarray(image_pil)
            
        raw_grad_heatmaps, filtered_grad_heatmaps = self.gradmap4vadObj.inference(image_pil, 
                                                self.anml_text_prompt_embedding[:, ad_class_names_id_topk].unsqueeze(1), 
                                                grad_mode=kwargs.pop("grad_mode", "grad_eclip"),
                                                # vis_resize_wh=vis_resize_wh,
                                                map_score_thr_dict=map_score_thr_dict, 
                                                **kwargs
                                            )
        fig = self.visualize_grad_cam_per_image(image_pil, filtered_grad_heatmaps, 
                                            fig_titles=None, 
                                            fig_scores=None, 
                                            map_score_thr_dict=map_score_thr_dict, 
                                            verbose=False, 
                                            fix_label_rec=True
                                            )
        
        if isget_map:
            return fig, filtered_grad_heatmaps
        return fig

 
class GRAD4VAD_XAI_PPA():
    
    def __init__(self, xai_clip_modelname: str = "ViT-L/14",  device: Literal["cuda", "cpu"] = "cuda", **kwargs):
    
        # setup models
        ADXAIObj = kwargs.get('ADXAIObj', None)
        if ADXAIObj is None:
            self.ADXAIObj = ADXAIDetector(xai_clip_modelname=xai_clip_modelname, 
                                        pooling_method="max", 
                                        isreturn_max_indices=True,
                                        device=device,
                                        **kwargs
                                     )
        else:
            self.ADXAIObj = ADXAIObj
            self.ADXAIObj.isreturn_max_indices = True
            self.ADXAIObj.pooling_method = "max"

        self.model =  ADXAIMAPDetector(objVADXAIModel=self.ADXAIObj, device=device)

        # setup target anomaly class settings
        self.xai_class_name_mapper = dict(zip(self.model.anml_class_names, range(len(self.model.anml_class_names))))
    
    def get_sliding_windows_no_padding(self, grid: list[int], window_shape: tuple[int], step_size: tuple[int]):
        """
        Slices a 2D/3D array into windows WITHOUT padding.
        Ensures full coverage by shifting the last windows backward
        so they align with the image boundaries.

        :param grid: numpy array (H, W) or (H, W, C)
        :param window_shape: (height, width)
        :param step_size: (step_y, step_x)
        """

        if isinstance(grid, list):
            grid = np.array(grid, dtype='uint8')

        img_h, img_w = grid.shape[:2]
        win_h, win_w = window_shape
        win_h, win_w = min(win_h, img_h), min(win_w, img_w)  # Ensure window isn't larger than image
        step_y, step_x = step_size
        step_y, step_x = max(1, step_y), max(1, step_x)  # Ensure step size is at least 1
        step_y, step_x = min(step_y, img_h), min(step_x, img_w)  # Ensure step isn't larger than window

        windows = []
        positions = []

        # Generate starting positions
        y_positions = list(range(0, img_h - win_h + 1, step_y))
        x_positions = list(range(0, img_w - win_w + 1, step_x))

        # Backward correction (ensure last window hits boundary)
        if y_positions[-1] != img_h - win_h:
            y_positions.append(img_h - win_h)

        if x_positions[-1] != img_w - win_w:
            x_positions.append(img_w - win_w)

        # Extract windows
        for y in y_positions:
            for x in x_positions:
                windows.append(grid[y:y + win_h, x:x + win_w])
                positions.append((y, x))
        
        print(f"patch extraction: input: {grid.shape}, window_shape: {window_shape}, step_size: {step_size}, n_pathes: {len(windows)}")
        return windows, positions, grid.shape

    def reconstruct_from_windows_no_padding(self, windows: list[np.ndarray], positions: list[tuple[int, int]], original_shape: tuple[int], window_shape: tuple[int], merge_mode="average"):
        """
        Reconstructs the image from windows using explicit positions (no padding).

        :param windows: list of window arrays
        :param positions: list of (y, x) positions corresponding to each window
        :param original_shape: shape of the original image (H, W) or (H, W, C)
        :param window_shape: (height, width)
        :param merge_mode: "average" or "max"
        """

        win_h, win_w = window_shape

        reconstructed = np.zeros(original_shape, dtype=np.float32)
        counts = np.zeros(original_shape, dtype=np.float32)

        if merge_mode == "average":
            for window, (y, x) in zip(windows, positions):
                reconstructed[y:y + win_h, x:x + win_w] += window
                counts[y:y + win_h, x:x + win_w] += 1

            # Avoid division by zero
            reconstructed = np.divide(
                reconstructed,
                counts,
                out=np.zeros_like(reconstructed),
                where=counts != 0
            )

        else:  # max merge
            reconstructed = np.full(original_shape, -np.inf, dtype=np.float32)

            for window, (y, x) in zip(windows, positions):
                reconstructed[y:y + win_h, x:x + win_w] = np.maximum(
                    reconstructed[y:y + win_h, x:x + win_w],
                    window
                )

            reconstructed[reconstructed == -np.inf] = 0

        return reconstructed

    def resize_bbox_xywh(self, box, orig_size, new_size):
        """
        Rescale a bounding box from original image size to new size.

        :param box: (x, y, w, h)
        :param orig_size: (orig_W, orig_H)
        :param new_size: (new_W, new_H)
        :return: resized box (x, y, w, h)
        """
        x, y, w, h = box
        orig_W, orig_H = orig_size
        new_W, new_H = new_size

        scale_x = new_W / orig_W
        scale_y = new_H / orig_H

        x_new = x * scale_x
        y_new = y * scale_y
        w_new = w * scale_x
        h_new = h * scale_y

        return (x_new, y_new, w_new, h_new)

    def draw_bbox_cv2(self, image, box, color=(0, 255, 0), thickness=2):
        """
        Draws a rectangle on an image using OpenCV.

        :param image: input image (numpy array)
        :param box: (x, y, w, h)
        :param color: box color (B, G, R)
        :param thickness: line thickness
        :return: image with box
        """
        x, y, w, h = map(int, box)

        top_left = (x, y)
        bottom_right = (x + w, y + h)

        img_out = image.copy()
        cv2.rectangle(img_out, top_left, bottom_right, color, thickness)

        return img_out

    def heatmap_to_boxes_fast(self, graymap, threshold=0.5, pool_kernel=3, class_id=0, ismultibox=False, around_peek_box_size=[10, 10], multi_box_alg="peak"):
        """
        graymap: (H, W, 3) tensor (on GPU)
        returns: list of boxes [x1, y1, x2, y2]
        """
        def rgb_to_gray(heatmap):
            """
            heatmap: (H, W, 3) or (3, H, W)
            returns: (H, W)
            """
            if heatmap.shape[0] == 3:
                r, g, b = heatmap[0], heatmap[1], heatmap[2]
            else:
                r, g, b = heatmap[..., 0], heatmap[..., 1], heatmap[..., 2]

            gray = 0.299 * r + 0.587 * g + 0.114 * b
            return gray
        
        def boxes_to_yolo(graymap, boxes, class_id=0):
            """
            graymap: (H, W) tensor with values in [0, 1] (confidence heatmap)
            boxes: list of [x1, y1, x2, y2]
            returns: Tensor [N, 6] -> (class, x_center, y_center, w, h, conf)
            """

            def xyxy_to_yolo(x1, y1, x2, y2, W, H):
                xc = (x1 + x2) / 2 / W
                yc = (y1 + y2) / 2 / H
                w = (x2 - x1) / W
                h = (y2 - y1) / H
                return xc, yc, w, h

            if len(boxes) == 0:
                # return torch.zeros((1, 6))  # no boxes, return empty tensor
                return []

            H, W = graymap.shape[0], graymap.shape[1]
            # --- Step 3: Build YOLO boxes ---
            yolo_boxes = []
            for box in boxes:
                xc, yc, w, h = xyxy_to_yolo(*box, W, H)
                score = graymap[box[1]:box[3], box[0]:box[2]].max().item()  # confidence from heatmap intensity
                # print("score: ", score)
                yolo_boxes.append(torch.tensor([
                                                class_id,          # class_id (single-class)
                                                xc, yc,
                                                w, h,
                                                np.round(score, 5)   # confidence
                                            ]))

            return torch.stack(yolo_boxes)

        def masks_to_boxes(mask):
            # Using connected components
            labeled, num = ndi.label(mask)
            # print(f"Number of connected components found: {num}")
            boxes = []
            for i in range(1, num+1):
                ys, xs = (labeled == i).nonzero()
                # print(f"Component {i}: Number of pixels: {len(xs)}, xs: {xs}, ys: {ys}")
                x1, x2 = xs.min(), xs.max()
                y1, y2 = ys.min(), ys.max()
                # print(f"x1: {x1}, x2: {x2}, y1: {y1}, y2: {y2}")
                if x2 > x1 and y2 > y1:  # valid box
                    boxes.append([x1, y1, x2, y2])
                # else:
                    # print("invalid box, skipping")
            return boxes

        if graymap.ndim == 3:
            graymap = rgb_to_gray(graymap)
        
        # print(f"graymap: min: {graymap.min()}, max: {graymap.max()}")
        graymap /= 255.0
        graymap = torch.round(graymap, decimals=5)
        
        # print(graymap.shape)
        # (H, W)
        # Step 1: threshold
        # adaptive threshold (better than fixed)
        # threshold = gray.mean() + 0.5 * gray.std()
        print(f"graymap: min: {graymap.min()}, max: {graymap.max()}, threshold: {threshold}")
        mask = graymap > threshold
        # print(f"mask: {mask.sum()}")
        # flat = gray.flatten()
        # topk = torch.topk(flat, k=3)
        
        # print(f"threshold: {threshold}, Number of graymap found: {graymap.sum().item()}, {graymap.max()}")
        if mask.sum() == 0:
            print(f"empty mask: {mask.sum()}")
            return boxes_to_yolo(graymap, [], class_id=class_id)
        
        if multi_box_alg == "peak":
            # Step 2: find local maxima (fast NMS-style)
            pooled = F.max_pool2d(graymap.unsqueeze(0).unsqueeze(0),
                                kernel_size=pool_kernel,
                                stride=1,
                                padding=pool_kernel//2)
            pooled = torch.round(pooled, decimals=5)
            peaks = (graymap == pooled.squeeze()) & mask

            # print(f"threshold: {threshold}, Number of peaks found: {peaks.sum().item()}, {pooled.max()}")
            # Step 3: get coordinates
            ys, xs = peaks.nonzero(as_tuple=True)

            if len(xs) > 0:
                if ismultibox:
                    boxes = []
                    for x, y in zip(xs, ys):
                        # small fixed-size box (fast approximation)
                        h, w = around_peek_box_size[0], around_peek_box_size[1] # create a 10x10 box around the peak
                        boxes.append([x.item()-w//2, y.item()-h//2,
                                    x.item()+w//2, y.item()+h//2])
                else:  
                    # single box around all peaks (slower but better for large connected regions)  
                    x1, x2 = xs.min(), xs.max()
                    y1, y2 = ys.min(), ys.max()
                    boxes = [(x1, y1, x2, y2)]
            else:
                boxes = []
        elif multi_box_alg == "connected":
            # Using connected components
            boxes = masks_to_boxes(mask)   
        
        # print(boxes, graymap[y1:y2, x1:x2].max().item())
        boxes = boxes_to_yolo(graymap, boxes, class_id=class_id)
        return boxes

    def to_yolo_results(self, image, yolo_boxes, class_id_name_mapper={0: "gun"}):
        """
        image: numpy array [HxWx3]
        yolo_boxes: Tensor [N,6]
        """

        H, W = image.shape[:2]

        # Convert to xyxy format (YOLO internal)
        boxes_xyxy = []
        for box in yolo_boxes:
            class_id, xc, yc, w, h, score = box.tolist()
            x1 = int((xc - w/2) * W)
            y1 = int((yc - h/2) * H)
            x2 = int((xc + w/2) * W)
            y2 = int((yc + h/2) * H)
            # print("conf: ", conf)
            boxes_xyxy.append([x1, y1, x2, y2, score, class_id])
        
        boxes_xyxy = torch.tensor(boxes_xyxy)

        results = Results(
            orig_img=image,
            path="",
            names=class_id_name_mapper,
            boxes=boxes_xyxy if len(boxes_xyxy) > 0 else None
        )

        return results

    def get_max_area_box(self, results, mms_threshold=0.001, box_peek_format="max_area", **kwargs):
        def nms_mms_combnine_boxes(results, nms_threshold=0.01, mms_threshold=0.01):
            detections = sv.Detections.from_ultralytics(results)
            # Apply Non-Max Merge (NMM) to combine boxes
            detections = detections.with_nmm(threshold=mms_threshold)
            #  Apply NMS to remove overlapping boxes (higher threshold = less strict)
            return detections

        detections = nms_mms_combnine_boxes(results, nms_threshold=0.5, mms_threshold=mms_threshold)
        boxes_xyxy = []
        # print("detections: ", len(detections.xyxy),  detections)
        
        if box_peek_format == "max_area":
            print("using max_area box peek format")
            max_area = 0
            for i in range(len(detections.xyxy)):
                box = detections.xyxy[i]
                box_area = (box[2] - box[0]) * (box[3] - box[1])*detections.confidence[i]
                if box_area > max_area:
                    max_area = box_area
                    boxes_xyxy = [(box[0], box[1], box[2], box[3], detections.confidence[i], detections.class_id[i])]
        else:  
            print("using sorted confidence format")
            for i in range(len(detections.xyxy)):
                box = detections.xyxy[i]
                boxes_xyxy.append((box[0], box[1], box[2], box[3], detections.confidence[i], detections.class_id[i]))
            boxes_xyxy = sorted(boxes_xyxy, key=lambda x: x[4], reverse=True) # sort by confidence
        return boxes_xyxy
                    
    def get_xai_heatmap(self, image_pil, xai_class_idx: int, map_score_thr_dict: Dict[str, Union[int, float, None]], grad_mode: Literal["grad_eclip", "grad_cam"] = "grad_eclip", resize_wh: Union[List[int], Tuple[int]] = (448, 448), **kwargs):
        """
        Extract XAI heatmap suing gradeclip, gradcam, etc
        """
        
        if resize_wh is not None:
            if isinstance(image_pil, np.ndarray):
                image_pil = util.PIL.Image.fromarray(image_pil)
            image_pil.thumbnail(resize_wh, util.PIL.Image.Resampling.LANCZOS) # resizes an image to a specified maximum size while preserving its original aspect ratio
            image_arr = np.asarray(image_pil)
            imsize = image_pil.size
        elif not isinstance(image_pil, np.ndarray):
            image_arr = np.asarray(image_pil)
        else:
            image_arr = image_pil
        
        fig, filtered_grad_heatmaps = self.model.inference(image_arr, xai_class_idx, 
                                                             map_score_thr_dict=map_score_thr_dict, 
                                                            grad_mode=grad_mode, isget_map=True)
        if filtered_grad_heatmaps[0] is None:
            filtered_grad_heatmap = np.zeros_like(image_arr)
        else:
            filtered_grad_heatmap = filtered_grad_heatmaps[0]
            
        if resize_wh is not None: 
            fig = fig.resize(imsize)
            filtered_grad_heatmap = np.asarray(util.PIL.Image.fromarray(filtered_grad_heatmap).resize(fig.size))

        return fig, filtered_grad_heatmap
    
    def ppa_patch_gradmap_xai_focus(self, image_pil, xai_class_idx, map_score_thr_dict: Dict[str, Union[int, float, None]], resize_wh: Union[List[int], Tuple[int]] = (448, 448), patch_size: int = 448, step_size: int = 336, **kwargs):
        print("Running patch sliding window accuracy up...")
        grad_mode = kwargs.get("grad_mode", "grad_eclip")
        isplot = kwargs.get("isplot", False)
        
        imsize = image_pil.size
        print(f"imsize: {imsize}, resize_wh: {resize_wh}")
        if resize_wh is not None:
            image_pil.thumbnail(resize_wh, util.PIL.Image.Resampling.LANCZOS)
        
        image_arr = np.asarray(image_pil)
        print(f"image_arr: {image_arr.shape}")
        print(f"patch_size: {patch_size}")
        win_size = (patch_size, patch_size)
        steps = (step_size, step_size)
        patches, p_positions, _ = self.get_sliding_windows_no_padding(image_arr, win_size, steps)
        
        patched_results = util.multiprocessor_speedup(self.get_xai_heatmap, 
                                                        xai_class_idx,
                                                        input_data=patches, 
                                                        isget_map=True, resize_wh=None, 
                                                        map_score_thr_dict=map_score_thr_dict, 
                                                        grad_mode=grad_mode,
                                                        num_workers=len(patches)
                                                        )
        
        fig_patches, filtered_grad_heatmaps_patches = list(map(lambda x: x[0], patched_results)), list(map(lambda x: x[1], patched_results))        
        
             
        # Reconstruct back to the original 10x10
        original_size = image_arr.shape
        # fig = reconstruct_from_windows(fig_patches, p_shape, original_size, win_size, steps, merge_mode="max").astype("uint8")
        fig = self.reconstruct_from_windows_no_padding(fig_patches, p_positions, original_size, win_size, merge_mode="max").astype("uint8")
        fig = util.PIL.Image.fromarray(fig).resize(imsize)
        
        filtered_grad_heatmaps = self.reconstruct_from_windows_no_padding(filtered_grad_heatmaps_patches, p_positions, original_size, win_size, merge_mode="max").astype("uint8")
        filtered_grad_heatmap = np.asarray(util.PIL.Image.fromarray(filtered_grad_heatmaps).resize(imsize)) # asarray() make is immutable, so need copy
            
        return fig, filtered_grad_heatmap

    def ppa_patch_clipcat_gradmap_xai_focus(self, image_pil, xai_class_idx: int, map_score_thr_dict: Dict[str, Union[int, float, None]], resize_wh: Union[List[int], Tuple[int]] = (448, 448), patch_size: int = 448, step_size: int = 336, **kwargs):
        print(f"Running auto-focus ppa_patch_clipcat_gradmap_xai...")
        grad_mode = kwargs.get("grad_mode", "grad_eclip")
        
        image_arr = np.asarray(image_pil)
        fig = image_pil.copy()
        imsize = fig.size
        win_size = (patch_size, patch_size)
        steps = (step_size, step_size)
        _image_pil = image_pil.copy()
        if resize_wh is not None:
            _image_pil.thumbnail(resize_wh, util.PIL.Image.Resampling.LANCZOS) 
        _image_arr = np.asarray(_image_pil)
        patches, p_positions, _ = self.get_sliding_windows_no_padding(_image_arr, win_size, steps)
        similarity_scores = self.ADXAIObj.inference(patches)[-1]
        frame_idx_with_max_score_for_target_anml = similarity_scores[1][0, xai_class_idx]
        box_xywh = p_positions[frame_idx_with_max_score_for_target_anml][::-1] + win_size  
        # patch loc resize to original 
        orig_box_xywh = self.resize_bbox_xywh(box_xywh, tuple(_image_pil.size), tuple(image_pil.size))
        # print(f"orig_box_xywh: {orig_box_xywh}")
        box_max_area = [orig_box_xywh[0], orig_box_xywh[1], orig_box_xywh[0] + orig_box_xywh[2], orig_box_xywh[1] + orig_box_xywh[3]]
        box_max_area = list((map(int, box_max_area)))
        # boxes_xyxy = [box_max_area + list(boxes_xyxy_sliced[0])[4:]]
        boxes_xyxy_expaneded = box_max_area + [0.0, 0] # dummy box with the same size as patch, and confidence score from clip-cat similarity score, this is a hack to make it work with the rest of the code, but it can be improved by directly using the similarity score as confidence score without going through the get_boxes function, which is designed for gradcam heatmap based box generation, but it can still work as a coarse box to focus on the most promising region
        image_arr = np.asarray(image_pil)
        box_max_area = list((map(int, boxes_xyxy_expaneded)))
        box_max_area[4] = boxes_xyxy_expaneded[4] # score
        image_arr_sliced = image_arr[box_max_area[1]:box_max_area[3], box_max_area[0]:box_max_area[2]]
        
        fig_sliced, filtered_grad_heatmaps_sliced = self.get_xai_heatmap(image_arr_sliced, xai_class_idx,
                                                        isget_map=True, resize_wh=None, 
                                                        map_score_thr_dict=map_score_thr_dict, 
                                                        grad_mode=grad_mode)
        filtered_grad_heatmap = np.asarray(util.PIL.Image.fromarray(filtered_grad_heatmaps_sliced).resize(imsize))
        fig.paste(fig_sliced, (box_max_area[0], box_max_area[1]))
                        
        return fig, filtered_grad_heatmap
    
    @util.timer_eval
    @util.process_logger                    
    def predict(self, image_rgb: Union[np.ndarray, util.PIL.Image], xai_cat_name: str = "gun shooting", 
                     bbox_conf: float = 0.85, iou: float = 0.5, ismultibox: bool = False, 
                     grad_mode: Literal["grad_eclip", "grad_cam"] = "grad_eclip", 
                     map_score_thr_dict: Dict[str, Union[int, float, None]] = {"map_score_min": 0.10, "map_score_max": None, "map_norm_score_min": 0.10}, 
                     autofocus_method: Optional[Literal["ppa_patch_clipcat_gradmap_xai", "ppa_patch_gradmap_xai", "gradmap_only_xai"]] = "ppa_patch_clipcat_gradmap_xai", 
                     patch_size: int = 448, step_size: int = 336, image_resize_wh=None, return_bbox=False,
                     isplot: bool =True, 
                     **kwargs):
        
        """ get heatmap from gradcam
            convert heatmap to boxes
            return in yolo
        """

        print(f"config: {bbox_conf}, iou: {iou}")
        print("image_rgb: ", image_rgb.shape)
        
        assert xai_cat_name in self.xai_class_name_mapper, "The anomaly category is not present in the precomputed category embedding. "
        
        xai_class_idx = self.xai_class_name_mapper[xai_cat_name]
        print(f"xai_cat_name: {xai_cat_name}, xai_class_idx: {xai_class_idx}")

        image_pil = util.PIL.Image.fromarray(image_rgb) if isinstance(image_rgb, np.ndarray) else image_rgb # rgb

        if autofocus_method == "ppa_patch_clipcat_gradmap_xai":
            fig, filtered_grad_heatmap = self.ppa_patch_clipcat_gradmap_xai_focus(image_pil, xai_class_idx, 
                                                                                  map_score_thr_dict=map_score_thr_dict,
                                                                                  grad_mode=grad_mode, resize_wh=image_resize_wh, 
                                                                                  isplot=isplot)
                
        elif autofocus_method == "ppa_patch_gradmap_xai":
            fig, filtered_grad_heatmap = self.ppa_patch_gradmap_xai_focus(image_pil, xai_class_idx, 
                                                                            map_score_thr_dict=map_score_thr_dict,
                                                                            grad_mode=grad_mode,
                                                                            patch_size=patch_size, step_size=step_size, 
                                                                            resize_wh=image_resize_wh, 
                                                                            mms_threshold=0.0, isplot=isplot) # mms_threshold=0.0 merge all heatmaps)
        else:
            fig, filtered_grad_heatmap = self.get_xai_heatmap(image_pil, xai_class_idx, grad_mode=grad_mode,
                                                                map_score_thr_dict=map_score_thr_dict, 
                                                                isget_map=True, 
                                                                resize_wh=image_resize_wh, isplot=isplot
                                                                )    
        boxes_xyxy = []
        if return_bbox:
            yolo_boxes = self.heatmap_to_boxes_fast(torch.from_numpy(filtered_grad_heatmap), 
                                                    threshold=bbox_conf, pool_kernel=3, class_id=0, 
                                                    ismultibox=ismultibox, 
                                                    multi_box_alg="connected")
                        
            if len(yolo_boxes) > 0:
                results = self.to_yolo_results(np.asarray(fig), yolo_boxes, class_id_name_mapper={0: xai_cat_name})
                show_image(results.plot(), isplot=True)
                boxes_xyxy = self.get_max_area_box(results, **kwargs)
            else:
                print("No valid bbox detection!")
                     
        return boxes_xyxy, fig
                          