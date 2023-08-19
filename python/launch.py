import subprocess
import os
import sys
import importlib.util
import shlex
import platform
import json

import argparse


commandline_args = os.environ.get('COMMANDLINE_ARGS', "")
sys.argv += shlex.split(commandline_args)


# 创建解析器对象
parser = argparse.ArgumentParser(description="命令行参数示例")

# 添加参数
# parser.add_argument('--reinstall_chatglm_cpp', type=bool, help="重新安装chatglm_cpp")


# 解析命令行参数
args = parser.parse_args()

# 打印参数值
print("args:", args)



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
        supported_minors = [7, 8, 9, 10, 11]

    if not (major == 3 and minor in supported_minors):
        print(f"""
INCOMPATIBLE PYTHON VERSION

This program is tested with 3.10.6 Python, but you have {major}.{minor}.{micro}.
If you encounter an error with "RuntimeError: Couldn't install torch." message,
or any other error regarding unsuccessful package (library) installation,
please downgrade (or upgrade) to the latest version of 3.10 Python
and delete current Python and "venv" folder in WebUI's directory.

You can download 3.10 Python from here: https://www.python.org/downloads/release/python-3109/

{"Alternatively, use a binary release of WebUI: https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases" if is_windows else ""}

Use --skip-python-version-check to suppress this warning.
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
        run_pip(f"install chatglm-cpp-0.2.4.tar.gz", "chatglm_cpp")
 
    if not is_installed('uvicorn'):
        run_pip(f"install uvicorn -i https://pypi.tuna.tsinghua.edu.cn/simple", "uvicorn")

    if not is_installed("pydantic_settings"):
        run_pip(f"install pydantic-settings -i https://pypi.tuna.tsinghua.edu.cn/simple","pydantic_settings")
 

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
    print(f"Launching API server")
    import api
    api.start()


if __name__ == "__main__":
    prepare_environment()
    start()
