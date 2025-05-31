# KAN-BackEnd 项目

## 项目简介
本项目为 上海市宜居性系统后端，采用微服务架构，包含如下服务：
- **api_gateway**：API 网关，统一入口，转发请求
- **auth_service**：认证服务，用户注册、登录、JWT鉴权
- **common_db**：共享数据库与认证工具

## 主要技术栈
- Python 3.9
- FastAPI
- SQLAlchemy
- JWT（python-jose）
- Alembic（数据库迁移）
- SQLite（默认数据库）

## 环境依赖
所有依赖已在 `environment.yml` 文件中列出，使用 conda 管理环境。

## 安装与运行

### 1. 创建并激活环境
```bash
conda env create -f environment.yml
conda activate kan-backend
```
> 注：若使用 vscode，需要配置 `python解释器`
>
> 若需更新环境，则：
>
> ```bash
> conda env update -f environment.yml
> ```

### 2. 启动各服务（使用 uvicorn）
分别进入各服务目录，运行如下命令：

```bash
# 启动 api_gateway
cd api_gateway
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 启动 auth_service
cd ../auth_service
uvicorn main:app --host 0.0.0.0 --port 8001 --reload

# 启动 compare_service
cd ../compare_service
uvicorn main:app --host 0.0.0.0 --port 8002 --reload

# 启动 data_service
cd ../data_service
uvicorn main:app --host 0.0.0.0 --port 8003 --reload

```

### 3. 访问接口
- API 网关默认端口：8000
- 认证服务默认端口：8001
- 模型对比服务默认端口：8002
- 栅格数据服务默认端口：8003

## 目录结构
```
api_gateway/         # API 网关服务
auth_service/        # 认证服务
common_db/           # 共享数据库与工具
compare_service/     # 模型对比服务
data_service/        # 栅格数据服务
environment.yml      # 环境依赖文件
readme.md            # 项目说明
```

