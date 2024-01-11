# Local-LLM 本地大模型
人人可拥有的本地模型服务，目前支持chatglm2和Llama-2。

### 下载
[下载](https://github.com/AGIUI/Local-LLM/archive/refs/heads/main.zip)
解压后，放到本地电脑的任意目录里 xxxx/Local-LLM

### 安装python

* 推荐使用python3.10
* [python下载地址](https://www.python.org/ftp/python/3.10.0/)


### 下载模型
- 请下载 chatglm2-ggml-q4_0.bin 和 Chinese-Llama-2-7b-ggml-q4.bin
- 放到目录 Local-LLM/models/xxx.bin


- 下载： [百度网盘链接](https://pan.baidu.com/s/15QrZnZqDIhuSFSq_JN0kiQ) 提取码：como 


* 其他chatglm2模型请到 [huggingface下载](https://huggingface.co/Xorbits/chatglm2-6B-GGML) 。如果使用更高精度的模型，下载后需要修改 [api.py](./python//api.py) 和 [webui.py](./python/webui.py) 里对应的文件名。

### 启动

#### WebUI模式

* win 系统 双击 webui-win.bat 

* mac 打开终端 cd xxx/Local-LLM ，然后 ./webui-mac.sh 回车

### API模式
支持chatglm2和llama-2

##### window系统:
双击 api-win.bat 直接运行

##### mac系统:
* 打开终端 输入：
```
cd xxx/Local-LLM
```
* 然后 输入:
```
./api-mac.sh
```


##### 客户端填写：
LocalLLM 设置里的 API URL 地址 填写 http://127.0.0.1:8000 

##### 启动 Llama-2
只需要传参llama即可启动，例如：
```
./api-win.bat llama
```


##### 可能会碰到的问题

mac提示没有app.sh权限，输入：
```
chmod 777 app.sh
```

mac pip 安装提示：
```
ERROR: chatglm_cpp-0.2.5-cp38-cp38-macosx_11_1_x86_64.whl is not a supported wheel on this platform.
```

原因：pip认为big sur是macOS_10_9。将所有捆绑的whl文件重命名为macos_10_9然后它就可以了

mac 查看线程数：
```
sysctl hw.logicalcpu
hw.logicalcpu: 8
```

[pyinstall 打包后运行提示第三方库找不到](https://blog.csdn.net/ldg513783697/article/details/119762461)

打开pyinstaller的目录，在hooks目录下创建文件，文件名一定是hook-第三方库.py，比如我用的eventlet库，我就需要创建hook-eventlet.py。
在该文件中添加如下代码：
```
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports =collect_all('eventlet') #collect_all的参数一定也是第三方库名，不要写错。
```
然后在用pyinstaller -F 打包exe,运行就不会报错。 



#### embedding
用llama_cpp实现

#### win打包
venv/Scripts/python -s -m pip install pyinstaller -i https://pypi.tuna.tsinghua.edu.cn/simple

pyinstaller -F app.py --clean

把models拷贝到 dist/目录下，运行 app.exe 启动

> app.exe port=8233 model=xxx max_tokens=2048 max_context_length=2048

## mac
python -m venv venv
source venv/bin/activate
pip install fastapi
pip install pydantic_settings
pip install sse_starlette
pip install chatglm_cpp-0.3.0-cp310-cp310-macosx_10_9_x86_64.whl

pyinstaller -F app.py --clean

把models拷贝到 dist/目录下，运行 app 启动


##### 感谢开源项目：

[chatglm.cpp](https://github.com/li-plus/chatglm.cpp)
[llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
[Xorbits](https://huggingface.co/Xorbits/chatglm2-6B-GGML)
