#!/bin/bash
# 启动FastAPI服务的脚本，需在linux加执行权限：chmod +x fastapi_up.sh，运行：./fastapi_up.sh

# 激活conda环境
source ~/anaconda3/bin/activate zimage || { echo "无法激活conda环境"; exit 1; }

# 杀死8888端口进程
lsof -ti:8888 | xargs -r kill -9

# 启动服务
nohup python fastapi_server.py > fastapi.log 2>&1 &

echo "服务已启动，PID: $!"
echo "日志文件：fastapi.log"