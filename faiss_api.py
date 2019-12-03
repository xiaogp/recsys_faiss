from flask import Flask, request
import json

import faiss

from utils import load_yaml_config

config = load_yaml_config("./config.yml")
faiss_model_path = config["model"]["faiss_model_path"]
faiss_topn = config["model"]["faiss_topn"]
index = faiss.read_index(faiss_model_path)

app = Flask(__name__)


@app.route("/faiss/similar_items/", methods=["GET"])
def check():
    # 默认返回内容
    return_dict = {'code': '200', 'msg': '处理成功', 'result': False}
    # 判断入参是否为空
    if request.args is None:
        return_dict['return_code'] = '504'
        return_dict['return_info'] = '请求参数为空'
        return json.dumps(return_dict, ensure_ascii=False)
    # 获取传入的参数
    get_data = request.args.to_dict()
    spu = int(get_data.get('spu'))
    # 对参数进行操作
    return_dict['result'] = faiss_search(spu)

    return json.dumps(return_dict, ensure_ascii=False)


# 功能函数
def faiss_search(spu):
    try:
        # 通过spu重建向量
        D, I = index.search(index.reconstruct(spu).reshape(1, -1), faiss_topn)
        result = {spu_index: score for spu_index, score in zip(I.tolist()[0], D.tolist()[0]) if spu_index != spu}
    except Exception as e:
        result = {"1": 1, "2": 2}
    return result


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000)
