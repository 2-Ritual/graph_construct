# graph_construct
https://github.com/openai/openai-cookbook在TuGraph上的实现
1. 环境要求python>=3.12
2. `pip install -r requirements.txt`安装依赖
3. `python all.py`运行完整代码
4. `.env`中配置`api_key`，默认使用百炼平台api，如果需要使用别的api，需要修改`config.py`中`BASE_URL`
5. `TuGraph`用户名、密码、数据库设置在`tugraph_api.py`中修改，`db_interface.py`只是接口
6. 数据加载路径在`utils.py`中修改`load_transcripts_from_pickle`修改`directory_path`参数
7. `prompts.py`中只设计了zero-shot版本的提示词，如果需要few-shot，需要在提示词中添加合适的示例
