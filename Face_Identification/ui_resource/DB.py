# 1. 定义连接池工具类
from dbutils.pooled_db import PooledDB  # 需要安装库 pip install DBUtils
import pymysql


class SQLPool:
    _pool = None  # 连接池单例

    def __init__(self):
        # 初始化连接池（仅在第一次实例化时创建）
        if not SQLPool._pool:
            SQLPool._pool = PooledDB(
                creator=pymysql,  # 数据库驱动
                maxconnections=5,  # 最大连接数
                mincached=2,  # 最小空闲连接
                maxcached=3,  # 最大空闲连接
                host="localhost",
                port=3306,
                user="root",
                passwd="wcm123456",
                database="faceidentity",
                charset="utf8",
                # 结果以字典的形式返回,字典的键就是字段名 目的:方便数据的操作
                cursorclass=pymysql.cursors.DictCursor
            )
        self.conn = None  # 每个实例持有独立连接
        self.cursor = None

    def get_conn(self):
        """获取连接（主窗口/子窗口调用此方法获取独立连接）"""
        self.conn = SQLPool._pool.connection()
        self.cursor = self.conn.cursor(pymysql.cursors.DictCursor)
        return self

    def close(self):
        """释放连接（放回池，而非真正关闭）"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()  # 连接池会自动回收

