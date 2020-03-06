import faiss
import numpy as np
from gensim.models import FastText
from sklearn.preprocessing import normalize

from utils import load_yaml_config, connect_mysql


def create_fasttext_model(sentence_list, model_path, embedding_path, mode="train", min_count=1,
                          size=128, sg=1, epochs=5):
    if mode == "train":
        model = FastText(min_count=min_count, size=size, sg=sg)
        model.build_vocab(sentence_list)
        model.train(sentence_list, total_examples=model.corpus_count, epochs=epochs)
    elif mode == "update":
        model = FastText.load(model_path)
        model.build_vocab(sentence_list, update=True)
        model.train(sentence_list, total_examples=model.corpus_count, epochs=5)
    model.save(model_path)
    model.wv.save_word2vec_format(embedding_path, binary=False)

    return model


def create_item_embedding(model, sentence, size, weight_dict, add_random):
    raw = np.zeros(size)
    for word in sentence:
        if word in list(model.wv.vocab.keys()):
            raw += model.wv[word] * weight_dict[word.split("|")[0]]
    if add_random:
        raw = raw * 100 + np.random.random(size) / 100

    return normalize(raw.reshape(1, -1))[0]


def create_faiss_model(item_embedding, item_list, faiss_path, size=128, mode="train"):
    item_embedding = np.array(item_embedding, dtype=np.float32)
    ids = np.array(item_list).astype("int")
    if mode == "train":
        index = faiss.index_factory(size, "IVF100,Flat", faiss.METRIC_INNER_PRODUCT)
        index.nprobe = 20
        index.train(item_embedding)
        # 初始化make_direct_map，reconstruct 重建向量
        index.make_direct_map()
        index_id = faiss.IndexIDMap2(index)
    elif mode == "update":
        index_id = faiss.read_index(faiss_path)
    index_id.add_with_ids(item_embedding, ids)
    # index保存
    faiss.write_index(index_id, faiss_path)

    return index


if __name__ == "__main__":
    config = load_yaml_config("./config.yml")
    # model
    fasttext_model_path = config["model"]["fasttext_model_path"]
    faiss_model_path = config["model"]["faiss_model_path"]
    fasttext_mode = config["model"]["fasttext_mode"]
    faiss_mode = config["model"]["faiss_mode"]
    embedding_size = config["model"]["embedding_size"]
    fasttext_epochs = config["model"]["fasttext_epochs"]
    fasttext_min_count = config["model"]["fasttext_min_count"]
    fasttext_sg = config["model"]["fasttext_sg"]
    faiss_topn = config["model"]["faiss_topn"]
    add_random = config["model"]["add_random"]
    # data
    embedding_path = config["data"]["embedding_path"]
    hbase_host = config["data"]["hbase_host"]
    hbase_port = config["data"]["hbase_port"]
    hbase_table = config["data"]["hbase_table"]
    mysql_host = config["data"]["mysql_host"]
    mysql_port = config["data"]["mysql_port"]
    mysql_db = config["data"]["mysql_db"]
    mysql_user = config["data"]["mysql_user"]
    mysql_password = config["data"]["mysql_password"]
    # weight
    PTY_NUM_1_weight = config["weight"]["PTY_NUM_1_weight"]
    PTY_NUM_2_weight = config["weight"]["PTY_NUM_2_weight"]
    PTY_NUM_3_weight = config["weight"]["PTY_NUM_3_weight"]
    PRODUCT_ORIGIN_weight = config["weight"]["PRODUCT_ORIGIN_weight"]
    BRAND_weight = config["weight"]["BRAND_weight"]
    # 创建权重字典
    weight_dict = {
        "PTY_NUM_1": PTY_NUM_1_weight,
        "PTY_NUM_2": PTY_NUM_2_weight,
        "PTY_NUM_3": PTY_NUM_3_weight,
        "PRODUCT_ORIGIN_NUM_ID": PRODUCT_ORIGIN_weight,
        "BRAND_ID": BRAND_weight
    }

    query = """select distinct ITEM_NUM_ID, PTY_NUM_1, PTY_NUM_2, PTY_NUM_3, PRODUCT_ORIGIN_NUM_ID, BRAND_ID 
               from goods_data where ITEM_NUM_ID is not null"""

    # 读取商品数据
    conn = connect_mysql(mysql_host, mysql_port, mysql_db, mysql_user, mysql_password)
    cursor = conn.cursor()
    cursor.execute(query)
    columns = ['ITEM_NUM_ID', 'PTY_NUM_1', 'PTY_NUM_2', 'PTY_NUM_3', 'PRODUCT_ORIGIN_NUM_ID', 'BRAND_ID']

    # 获得spu和spu的特征
    item_sentence_dict = {}
    for line in cursor.fetchall():
        item = line[0]
        sentence = [column + "|" + str(value) for column, value in zip(columns[1:], line[1:]) if value != None]
        item_sentence_dict.setdefault(item, [])
        item_sentence_dict[item].append(sentence)
    # 每一个item取文本最长的，同一个spu存在多条记录，特征无法去重复
    item_sentence_dict = {item: sorted(sentences, key=lambda x: len(x), reverse=True)[0] for item, sentences in
                          item_sentence_dict.items()}

    cursor.close()
    conn.close()

    # 得到item_List和sentence_list
    item_list, sentence_list = list(item_sentence_dict.keys()), list(item_sentence_dict.values())

    # 训练词向量
    fasttext_model = create_fasttext_model(sentence_list, fasttext_model_path, embedding_path,
                                           mode=fasttext_mode, min_count=fasttext_min_count,
                                           size=embedding_size, sg=fasttext_sg, epochs=fasttext_epochs)

    # 获得句子向量
    item_embedding = [create_item_embedding(fasttext_model, sentence, embedding_size, weight_dict, add_random) for
                      sentence in sentence_list]
    item_embedding = np.array(item_embedding, dtype=np.float32)
    # 训练faiss
    index = create_faiss_model(item_embedding, item_list, faiss_path=faiss_model_path, size=embedding_size,
                               mode=faiss_mode)




