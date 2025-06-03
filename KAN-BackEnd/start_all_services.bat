@echo off
REM 启动 KAN-BackEnd 各微服务（每个服务单独窗口）

start cmd /k "cd /d %~dp0api_gateway && conda activate kan-backend && uvicorn main:app --host 0.0.0.0 --port 8000 --reload"
start cmd /k "cd /d %~dp0auth_service && conda activate kan-backend && uvicorn main:app --host 0.0.0.0 --port 8001 --reload"
start cmd /k "cd /d %~dp0compare_service && conda activate kan-backend && uvicorn main:app --host 0.0.0.0 --port 8002 --reload"
start cmd /k "cd /d %~dp0data_service && conda activate kan-backend && uvicorn main:app --host 0.0.0.0 --port 8003 --reload"
start cmd /k "cd /d %~dp0predict_service && conda activate kan-backend && uvicorn main:app --host 0.0.0.0 --port 8004 --reload"

echo 所有服务已尝试启动（请检查各窗口输出）
pause 