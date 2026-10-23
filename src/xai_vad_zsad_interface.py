# =============================================================================
# XAVAD: Unified Explainable Anonymized Video Anomaly Detection (VAD) System
# =============================================================================
# This code defines a Python class for integrating the VLM into a XAVAD. 
# The class, ADXAIVLMInference, loads a specified VLM model variant, sets up preprocessing for prompts, and runs inference. 

# Author: Mulugeta W. Asres
# Email: muleina2000@gmail.com
# Date: October 2025
#####################################################################

import json
import sys, os
import numpy as np
from typing import List, Tuple, Dict, Union, Optional, Literal, Any
import cv2
import base64
from pydantic import BaseModel, Field, field_validator, ValidationError
from ollama import chat as ollama_chat
import utilities as util

OLLAMA_URL = "http://localhost:11434/api/chat"

class AnalysisScore(BaseModel):
    anomaly_status: str = Field(description="The alert status flag of anomaly detection")
    category: str = Field(description="The category of the analysis")
    reason: str = Field(max_length=300, description="The video caption and reason for the decision")

    @field_validator('reason')
    def check_review_reason(cls, v):
        if not v:
            raise ValueError('reason is required when anomaly_status is YES')
        return v
    
class ADXAIVLMInference():
        
    def __init__(self, **kwargs):
        self.model = kwargs.get("model", "blaifa/InternVL3_5:4B")
        self.keep_alive = kwargs.get("keep_alive", -1)  
        
        print("Loading VLM model...")
        ollama_chat(model=self.model, messages=[], keep_alive=self.keep_alive)
        
        self.set_config(**kwargs)
        print("*"*50)
    
    def set_config(self, **kwargs):
        self.num_frames = kwargs.get("num_frames", 8) 
        self.target_resolution = kwargs.get("target_resolution", (240, 240))
        self.prompt_json_filename = kwargs.get("prompt_json_filename", "pmt_v1.json")
        
        self.options = { 
                        # latency and accuracy trade-off
                        "num_ctx": kwargs.get("num_ctx", None),
                        # restrict to factual
                        "temperature": kwargs.get("temperature", 0),
                        "top_p": kwargs.get("top_p", 1),
                        "top_k": kwargs.get("top_k", 1),
                        "repeat_penalty": kwargs.get("repeat_penalty", 1.0),
                        # latency
                        "num_predict": kwargs.get("num_predict", 50)
                    }
        
        self.prompt_json = util.load_json(filename=self.prompt_json_filename, filepath="/home/mulugetawa/AI4CITIZENS/PACA/src/model/vlm")
        
        self.prompt_user_questions = self.prompt_json["prompt_user"]
        self.prompt_system = {"role": "system", "content": self.prompt_json["prompt_system"]}

        print(f"system prompt length: {len(self.prompt_system['content'])}")
        print(f"user prompt length: {len(self.prompt_user_questions)}")

        self.inference_func = self.ollama_query_vlm
                    
    def extract_and_encode_video_frames_base64(self, video_frames: List[np.ndarray], num_frames: Optional[int] = None, isb64encode: bool = True, **kwargs):
        
        total = len(video_frames)
        indices = np.linspace(0, total - 1, total if num_frames is None else min(total, num_frames)).astype(int)
        frames = []
        # print(f"isb64encode: {isb64encode}, video_frames: {video_frames[0].shape}, total: {total}, indices:{len(indices)}")
        
        for idx in indices:
            frame = video_frames[idx]

            if isb64encode:
                _, buf = cv2.imencode(".jpg", frame)
                frames.append(base64.b64encode(buf).decode("utf-8"))
            else:
                frames.append(frame) # if frame is filepath
        
        # print(f"frames: {len(frames)}, indices:{len(indices) }")
        return frames

    def prepare_payload(self, frame_list: List[np.ndarray], isb64encode: bool = True, **kwargs):
        payload = {
                    "model": self.model,
                    "messages": [
                                    self.prompt_system,
                                        {
                                            "role": "user",
                                            "content": self.prompt_user_questions,
                                            "images": self.extract_and_encode_video_frames_base64(frame_list, 
                                                                                                  isb64encode=isb64encode, 
                                                                                                  **{
                                                                                                      **dict(num_frames=self.num_frames, target_resolution=self.target_resolution), 
                                                                                                      **kwargs
                                                                                                  })
                                        }
                                ],
                    "options": self.options,
                    "stream": False,
                    # 'think': False,  # Forcibly disables the thinking token delay
                    "format": AnalysisScore.model_json_schema() # Force JSON structure       
                }

        return payload
    
    def ollama_query_vlm(self, *args, **kwargs):
        vlm_response = None
        try:
            payload = self.prepare_payload(*args, **kwargs)
            response = ollama_chat(**payload)
            raw_response = response.message.content.strip()
            # print("VLM: ", raw_response)
            if raw_response is None or len(raw_response) == 0:
                return raw_response
        
            # Strip out the entire thought process
            vlm_response = util.re.sub(r'<thought>.*?</thought>', '', raw_response, flags=util.re.DOTALL).strip()
            # Defensive strip against accidental markdown formatting
            vlm_response = util.re.sub(r'^```(?:json)?\s*|```$', '', vlm_response, flags=util.re.IGNORECASE | util.re.MULTILINE).strip()
            vlm_response = vlm_response.replace("```json", "").replace("```", "").strip()
            
            # print("VLM (after cleaning): ", vlm_response)
            vlm_response = util.json.loads(vlm_response)
            # Fallback mapping if Qwen hallucinates 'caption' instead of 'reason'
            if "caption" in vlm_response and "reason" not in vlm_response:
                vlm_response["reason"] = vlm_response.pop("caption")

            vlm_response = AnalysisScore.model_validate(vlm_response)
            # print("Success! Parsed Schema:")
            vlm_response = vlm_response.model_dump()

        except (ValidationError, json.JSONDecodeError) as ex:
            # print(f"Extraction failed. Raw output received:\n{raw_response}")
            # print(f"Validation Error: {ex}")
            try:
                # print("truncation check: ", "{" in raw_response, " " , '"}' not in raw_response)
                if  "{" in vlm_response and not vlm_response.replace(" ", "").endswith('"}'):
                    if vlm_response.endswith('"'):
                        vlm_response += '}'
                    elif vlm_response.endswith('}'):
                        vlm_response = vlm_response[:-1]
                        vlm_response +='"}'
                    else:
                        vlm_response += '"}'
                    print("VLM (adjusted): ", vlm_response)
                    
                vlm_response = util.json.loads(vlm_response)
                # Fallback mapping if Qwen hallucinates 'caption' instead of 'reason'
                if "caption" in vlm_response and "reason" not in vlm_response:
                    vlm_response["reason"] = vlm_response.pop("caption")
            except:
                print("Failed to parse VLM response as JSON. Set response to None.")
                vlm_response = None
        return vlm_response

    @util.timer_eval
    @util.process_logger
    def inference(self, frame_list: List[np.ndarray], **kwargs):
        results = self.inference_func(frame_list, **kwargs)
        return results
      