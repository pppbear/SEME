@echo off
REM 

start cmd /k "cd /d %~dp0api_gateway && conda activate kan-backend && uvicorn main:app --host 0.0.0.0 --port 8000 --reload"
start cmd /k "cd /d %~dp0auth_service && conda activate kan-backend && uvicorn main:app --host 0.0.0.0 --port 8001 --reload"
start cmd /k "cd /d %~dp0compare_service && conda activate kan-backend && uvicorn main:app --host 0.0.0.0 --port 8002 --reload"
start cmd /k "cd /d %~dp0data_service && conda activate kan-backend && uvicorn main:app --host 0.0.0.0 --port 8003 --reload"
start cmd /k "cd /d %~dp0predict_service && conda activate kan-backend && uvicorn main:app --host 0.0.0.0 --port 8004 --reload"
start cmd /k "cd /d %~dp0analyze_service && conda activate kan-backend && uvicorn main:app --host 0.0.0.0 --port 8005 --reload"

echo 
pause 