from flask import Flask, request, jsonify
import faiss

from utils import load_yaml_config

config = load_yaml_config("./config.yml")
faiss_model_path = config["model"]["faiss_model_path"]
index = faiss.read_index(faiss_model_path)

app = Flask(__name__)


@app.route("/faiss/similar_items/", methods=["GET"])
def check():
    try:
        get_data = request.args.to_dict()
        spu = int(get_data.get('spu'))
        n_items = int(get_data.get('n_items'))

    except Exception as e:
        return jsonify({'code': 400, 'msg': '无效请求', 'trace': traceback.format_exc()}), 400

    result = faiss_search(spu, n_items)
    return jsonify({'code': 200, 'msg': '处理成功', 'result': result}), 200


# 功能函数
def faiss_search(spu, n_items):
    try:
        # 通过spu重建向量
        D, I = index.search(index.reconstruct(spu).reshape(1, -1), n_items + 1)
        # result = I.tolist()[0]
        result = {spu_index: round(score, 4) for spu_index, score in zip(I.tolist()[0], D.tolist()[0])
                  if spu_index != spu}
    except RuntimeError:
        result = {"1": 1}
    return result


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
