import datetime

import faiss
from fastapi import FastAPI, Query

from utils import load_yaml_config

config = load_yaml_config("./config.yml")
faiss_model_path = config["model"]["faiss_model_path"]
index = faiss.read_index(faiss_model_path)

model_update_time = None

app = FastAPI()


@app.get("/faiss/similar_items/")
async def get_single_item_similar(
        spu_id: int = Query(
                ..., 
                title="spu_id", 
                description="item spu id", 
                gt=0), 
        n_items: int = Query(
                9, 
                titel="n_items", 
                description="topN of similar items", 
                ge=1)
        ):
    tim_str = datetime.datetime.now().strftime("%Y%m%d")
    global model_update_time, index
    item_list = []
    
    try:
        if tim_str != model_update_time:
            model_update_time = tim_str
            # 重新读取模型文件
            index = faiss.read_index(faiss_model_path)
        distance, item_index = index.search(index.reconstruct(spu_id).reshape(1, -1), n_items + 1)
        item_list = [spu for spu in item_index.tolist()[0] if spu != spu_id]
        result = {"code": 200, "msg": "success", "res": item_list}
    except Exception as e:
        result = {"code": 400, "msg": e.args, "res": item_list}
        
    return result
