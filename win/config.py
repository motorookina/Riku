import os
from dotenv import load_dotenv

# 加载 .env 到环境变量
load_dotenv()

class Config:

    # LLM相关
    LLM_API_KEY=os.getenv("LLM_API_KEY")
    LLM_BASE_URL=os.getenv("LLM_BASE_URL")
    MODEL=os.getenv("MODEL")

    # 数据库
    DB_USER=os.getenv("DB_USER")
    DB_PASSWD=os.getenv("DB_PASSWD")
    DB_HOST=os.getenv("DB_HOST")
    DB_PORT=os.getenv("DB_PORT")
    DB_NAME=os.getenv("DB_NAME")

    # 认证
    JWT_SECRET_KEY=os.getenv("JWT_SECRET_KEY")
    ACCESS_TOKEN_EXPIRE_MINUTES=os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES")

    # 数据源
    # tushare
    TUSHARE_TOKEN=os.getenv("TUSHARE_TOKEN")

    # 理性仁API
    LIXINGER_TOKEN=os.getenv("LIXINGER_TOKEN")

    # docker容器
    _DEFAULT_IMAGE=os.getenv("_DEFAULT_IMAGE")
    _DEFAULT_WORKING_DIR=os.getenv("_DEFAULT_WORKING_DIR")
    _DEFAULT_EXECUTE_TIMEOUT=os.getenv("_DEFAULT_EXECUTE_TIMEOUT")
    _DEFAULT_MAX_OUTPUT_BYTES=os.getenv("_DEFAULT_MAX_OUTPUT_BYTES")
    _DEFAULT_CONTAINER_NAME=os.getenv("_DEFAULT_CONTAINER_NAME")


# 实例化，供其他地方导入
config = Config()