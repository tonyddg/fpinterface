from typing import List
from typing_extensions import Annotated
from pydantic import BaseModel, Field

import numpy as np
from fastapi import FastAPI, UploadFile, HTTPException
import io
import logging

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.server_utility import FPRespone, FPServer, BboxModel

model = FPServer(os.environ.get("SERVER_CONFIG", "/app/app/server_config.yaml"))
app = FastAPI()

@app.post("/infer", response_model = FPRespone)
async def infer(target: str, is_bbox_pose: bool, pack_file: UploadFile):
    '''
    上传传感数据进行推理
    '''
    
    raw_byte = await pack_file.read()

    try:
        raw_npz = np.load(io.BytesIO(raw_byte), allow_pickle = False)
        pack_arr = raw_npz["pack_arr"]
        result = model.infer(target, is_bbox_pose, pack_arr)
    
    except Exception as e:
        logging.error(f"Raise expect {e}")
        raise HTTPException(400, f"Raise expect {e}")

    logging.info(f"Respond result: {result}")
    return result

class CamKModel(BaseModel):
    cam_k: List[float] = Field(min_length = 9, max_length = 9)
@app.post("/set_cam_k")
async def set_cam_k(body: CamKModel):
    '''
    设置相机内参
    '''

    try:
        model.set_k(body.cam_k)
    except Exception as e:
        logging.error(f"Raise expect {e}")
        raise HTTPException(400, f"Raise expect {e}")

    logging.info(f"Set cam K to : {body.cam_k}")

@app.get("/get_cfg")
async def get_cfg():
    '''
    获取推理配置参数
    '''

    return model.cfg


@app.get("/get_mesh_bbox", response_model = BboxModel)
async def get_mesh_bbox(target: str):
    '''
    获取模型包容盒
    '''

    try:
        res = model.get_mesh_bbox(target)
    
    except Exception as e:
        logging.error(f"Raise expect {e}")
        raise HTTPException(400, f"Raise expect {e}")

    logging.info(f"Respond result: {res}")
    return res
