# 安装python
* 推荐使用python3.10


# 下载模型
默认使用 chatglm2-ggml-q4_0.bin，放到目录 Local-LLM/models/ 

链接：https://pan.baidu.com/s/1YVqaf2uXL73fTTzpab8tIQ 
提取码：como 


* 其他模型下载 https://huggingface.co/Xorbits/chatglm2-6B-GGML

* 模型下载后修改api.py里的文件名


| Name | Quant method | Bits | Size |
|------|--------------|------|------|
| chatglm2-ggml-q4_0.bin | q4_0 | 4 | 3.5 GB  |
| chatglm2-ggml-q4_1.bin | q4_1 | 4 | 3.9 GB  |
| chatglm2-ggml-q5_0.bin | q5_0 | 5 | 4.3 GB  |
| chatglm2-ggml-q5_1.bin | q5_1 | 5 | 4.7 GB  |
| chatglm2-ggml-q8_0.bin | q8_0 | 8 | 6.6 GB  |



# 启动
* win 系统 双击 app.bat 一键启动
* mac 打开终端 cd xxx/Local-LLM ，然后 ./app.sh 回车


* 运行成功后把 http://127.0.0.1:8000 填写到 ChatGPT设置里的 API URL 地址里



# 问题
mac提示没有app.sh权限
输入：chmod 777 app.sh

mac pip安装提示：
ERROR: chatglm_cpp-0.2.5-cp38-cp38-macosx_11_1_x86_64.whl is not a supported wheel on this platform.
原因：
pip认为big sur是macOS_10_9。我将所有捆绑的车轮文件重命名为macos_10_9然后它就工作了。

mac 查看：
sysctl hw.logicalcpu

hw.logicalcpu: 8

