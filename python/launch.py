import subprocess
import os
import sys
import importlib.util
import shlex
import platform
import json
from pathlib import Path

import argparse


commandline_args = os.environ.get('COMMANDLINE_ARGS', "")


# 打印参数值
print("args:", commandline_args)



python = sys.executable
git = os.environ.get('GIT', "git")
index_url = os.environ.get('INDEX_URL', "")
stored_commit_hash = None
skip_install = False
dir_repos = "repositories"

def is_installed(package):
    try:
        spec = importlib.util.find_spec(package)
    except ModuleNotFoundError:
        return False

    return spec is not None


def check_python_version():
    is_windows = platform.system() == "Windows"
    major = sys.version_info.major
    minor = sys.version_info.minor
    micro = sys.version_info.micro

    if is_windows:
        supported_minors = [10]
    else:
        supported_minors = [8, 9, 10]

    if not (major == 3 and minor in supported_minors):
        print(f"""
PYTHON 版本不匹配
你的 PYTHON 版本： {major}.{minor}.{micro}
本程序在 PYTHON 3.10 完成测试.
""")

def run(command, desc=None, errdesc=None, custom_env=None, live=False):
    if desc is not None:
        print(desc)

    if live:
        result = subprocess.run(command, shell=True, env=os.environ if custom_env is None else custom_env)
        if result.returncode != 0:
            raise RuntimeError(f"""{errdesc or 'Error running command'}.
Command: {command}
Error code: {result.returncode}""")

        return ""

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, env=os.environ if custom_env is None else custom_env)

    if result.returncode != 0:

        message = f"""{errdesc or 'Error running command'}.
Command: {command}
Error code: {result.returncode}
stdout: {result.stdout.decode(encoding="utf8", errors="ignore") if len(result.stdout)>0 else '<empty>'}
stderr: {result.stderr.decode(encoding="utf8", errors="ignore") if len(result.stderr)>0 else '<empty>'}
"""
        raise RuntimeError(message)

    return result.stdout.decode(encoding="utf8", errors="ignore")

def commit_hash():
    global stored_commit_hash

    if stored_commit_hash is not None:
        return stored_commit_hash

    try:
        stored_commit_hash = run(f"{git} rev-parse HEAD").strip()
    except Exception:
        stored_commit_hash = "<none>"

    return stored_commit_hash

def run_pip(args, desc=None):
    if skip_install:
        return

    index_url_line = f' --index-url {index_url}' if index_url != '' else ''
    return run(f'"{python}" -m pip {args} --prefer-binary{index_url_line}', desc=f"Installing {desc}", errdesc=f"Couldn't install {desc}")



def prepare_environment():
    global skip_install

    # torch_command = os.environ.get('TORCH_COMMAND', "pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117")
    # requirements_file = os.environ.get('REQS_FILE', "requirements.txt")

    # run_pip(f"install insightface==0.7.3 -i https://pypi.tuna.tsinghua.edu.cn/simple", "insightface")
    # run_pip(f"install onnxruntime==1.15.0 -i https://pypi.tuna.tsinghua.edu.cn/simple", "onnxruntime")

    #!!!直接使用ggml文件不需要安装torch
    # if not is_installed("torch"):
    #     run_pip(f"install torch>=2.0 -i https://pypi.tuna.tsinghua.edu.cn/simple","torch")

    # if not is_installed("transformers"):
    #     run_pip(f"install transformers==4.30.2 -i https://pypi.tuna.tsinghua.edu.cn/simple","transformers")
    
    # # if not is_installed("accelerate"):
    # #     run_pip(f"install accelerate -i https://pypi.tuna.tsinghua.edu.cn/simple","accelerate")

    # if not is_installed("tabulate"):
    #     run_pip(f"install tabulate -i https://pypi.tuna.tsinghua.edu.cn/simple","tabulate")
  
    # os.environ['DGGML_CUBLAS']="ON"    
    if not is_installed('chatglm_cpp'):
        #安装编译好的安装包
        if sys.platform.startswith("win"):
            run_pip(f"install chatglm_cpp/chatglm_cpp-0.3.0-cp310-cp310-win_amd64.whl", "chatglm_cpp")
        # elif sys.platform.startswith("darwin"):
        #     run_pip(f"install chatglm_cpp/chatglm_cpp-0.3.0-cp310-cp310-macosx_10_9_x86_64.whl", "chatglm_cpp")
        #自行编译
        # run_pip(f"install chatglm_cpp/chatglm-cpp-0.3.0.tar.gz", "chatglm_cpp")
    
    if not is_installed("llama_cpp"):
        run_pip(f"install llama-cpp-python[server] -i https://pypi.tuna.tsinghua.edu.cn/simple", "llama_cpp")

    if not is_installed('uvicorn'):
        run_pip(f"install uvicorn -i https://pypi.tuna.tsinghua.edu.cn/simple", "uvicorn")

    if not is_installed('fastapi'):
        run_pip(f"install fastapi -i https://pypi.tuna.tsinghua.edu.cn/simple", "fastapi")

    if not is_installed('sse_starlette'):
        run_pip(f"install sse-starlette -i https://pypi.tuna.tsinghua.edu.cn/simple", "sse_starlette")

    if not is_installed("pydantic_settings"):
        run_pip(f"install pydantic-settings -i https://pypi.tuna.tsinghua.edu.cn/simple","pydantic_settings")
 
    if not is_installed("gradio"):
        run_pip(f"install gradio -i https://pypi.tuna.tsinghua.edu.cn/simple","gradio")

    check_python_version()

    commit = commit_hash()

    print(f"Python {sys.version}")
    print(f"Commit hash: {commit}")

    print(platform.system(),platform.python_version().startswith("3.10"))
    
    

    # 安装torch GPU版本
    # if args.reinstall_torch or not is_installed("torch") or not is_installed("torchvision"):
    #     run(f'"{python}" -m {torch_command}', "Installing torch and torchvision", "Couldn't install torch", live=True)


    # 安装相关依赖
    # if not os.path.isfile(requirements_file):
    #     requirements_file =os.path(__file__,requirements_file) 
    # # print(requirements_file)
    # #  -i https://pypi.tuna.tsinghua.edu.cn/simple
    # run_pip(f"install -r \"{requirements_file}\"", "requirements for Web UI")


 


def start():
    # print(commandline_args.lower()=='--llama')
    if 'llama' in commandline_args.lower():
        # 使用Llama-2 的api模式
        print("使用Llama-2 的api模式")
        DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models/Chinese-Llama-2-7b-ggml-q4.bin"
        if not os.path.exists(DEFAULT_MODEL_PATH):
            print('##### 模型文件不存在：',DEFAULT_MODEL_PATH)
        
        os.system(f'"{python}" -m llama_cpp.server --model ./models/Chinese-Llama-2-7b-ggml-q4.bin')

    else:

        DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models/chatglm3-ggml-q4_0.bin"
    
        if not os.path.exists(DEFAULT_MODEL_PATH):
            print('##### 模型文件不存在：',DEFAULT_MODEL_PATH)

        if commandline_args=='--web':
            print("chatglm3 的webui模式")
            import webui
            webui.start()
        else:
            print("chatglm3 的api模式")
            #print(f"Launching API server")
            import api
            api.start()


if __name__ == "__main__":
    prepare_environment()
    start()
