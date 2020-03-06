import yaml

import happybase
import pymysql


def load_yaml_config(yaml_file):
    with open(yaml_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def connect_hbase(tablename, host, port):
    connection = happybase.Connection(host, port)
    table = connection.table(tablename)
    return table


def connect_mysql(host, port, db_name, user, password):
    db_config = {"host": host, "port": port, "database": db_name,
                 "user": user, "password": password}
    conn = pymysql.connect(**db_config)
    return conn

