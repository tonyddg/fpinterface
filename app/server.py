import numpy as np
from fastapi import FastAPI, UploadFile, HTTPException
import io
import logging

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.server_utility import FPRespone, FPServer

model = FPServer(os.environ.get("SERVER_CONFIG", "/app/app/server_config.yaml"))
app = FastAPI()

@app.post("/infer", response_model = FPRespone)
async def infer(target: str, pack_file: UploadFile):
    '''
    上传传感数据进行推理
    '''
    
    raw_byte = await pack_file.read()

    try:
        raw_npz = np.load(io.BytesIO(raw_byte), allow_pickle = False)
        pack_arr = raw_npz["pack_arr"]
        result = model.infer(target, pack_arr)
    
    except Exception as e:
        logging.error(f"Raise expect {e}")
        raise HTTPException(400, f"Raise expect {e}")

    logging.info(f"Respond result: {result}")
    return result

@app.get("/get_cfg")
async def get_cfg():
    '''
    获取推理配置参数
    '''

    return model.cfg
