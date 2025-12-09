安装后台程序管理工具 `supervisor`：

```bash
pip install supervisor
```

编写配置，启动 `supervisord` 维持后台任务：

```bash
supervisord -c ./mcp_supervisor.conf
```

检查端口是否正常提供服务：

```bash
lsof -i :8000
lsof -i :8001
```

终止后台任务：

```bash
pkill -f supervisord
```
