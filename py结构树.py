import os
import sys

def generate_project_info(root_dir, output_filename="project_info.txt"):
    """
    生成项目信息报告，包括目录树和Python文件内容。
    该报告会过滤掉虚拟环境、缓存目录和常见的二进制文件，
    只保留对AI分析项目核心有用的信息。

    Args:
        root_dir (str): 要分析的根目录路径。
        output_filename (str): 报告输出的文件名。
    """
    try:
        # 确保根目录存在
        if not os.path.isdir(root_dir):
            print(f"错误：指定的路径不存在或不是一个目录: {root_dir}")
            return

        # 定义要忽略的目录和文件模式
        ignored_dirs = [
            'venv', '.venv', 'env', '__pycache__', '.git', '.idea',
            'node_modules', 'dist', 'build', '.vscode', '.pytest_cache',
            '__pycache__', 'site-packages' # 常见Python虚拟环境和缓存目录
        ]
        # 定义要忽略的文件扩展名（二进制文件、日志、配置文件等）
        ignored_extensions = [
            '.log', '.bin', '.dll', '.exe', '.so', '.dylib', '.DS_Store',
            '.pyc', '.pyo', '.ipynb_checkpoints', '.tmp', '.bak', '.swp',
            '.sqlite3', '.db', '.dat', '.json', '.yaml', '.yml', '.toml',
            '.xml', '.txt', '.md', '.csv', '.xlsx', '.xls', '.pdf',
            '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.ico',
            '.zip', '.tar', '.gz', '.rar', '.7z', '.mp3', '.mp4', '.avi',
            # 您可能需要根据具体项目进一步扩展此列表
        ]

        with open(output_filename, 'w', encoding='utf-8') as f:

            # === 1. 生成目录结构树 (仅核心文件夹和.py文件) ===
            f.write("=== 核心目录结构树 (仅文件夹和.py文件): ===\n")
            f.write("\n")

            # 存储所有.py文件的路径，以便后续读取内容
            py_files_paths = []

            for dirpath, dirnames, filenames in os.walk(root_dir):
                # 过滤掉要忽略的目录
                dirnames[:] = [d for d in dirnames if d not in ignored_dirs]

                # 检查当前目录是否为被忽略的目录本身
                if os.path.basename(dirpath) in ignored_dirs and dirpath != root_dir:
                    continue

                # 计算当前目录的深度，用于树形缩进
                depth = dirpath[len(root_dir):].count(os.sep)
                indent = '    ' * depth
                
                # 写入当前目录名
                if dirpath == root_dir:
                    f.write(f"{root_dir}\n")
                else:
                    f.write(f"{indent}|-- {os.path.basename(dirpath)}\n")

                # 写入 .py 文件 (过滤掉忽略的文件扩展名)
                for filename in sorted(filenames):
                    if filename.endswith(".py") and not any(filename.endswith(ext) for ext in ignored_extensions):
                        f.write(f"{indent}|   |-- {filename}\n")
                        py_files_paths.append(os.path.join(dirpath, filename))
            f.write("\n")

            # === 2. 列出所有核心Python文件 (.py) 列表 ===
            f.write("=== 核心Python文件 (.py) 列表 (绝对路径): ===\n")
            f.write("\n")
            if py_files_paths:
                for py_file_path in sorted(py_files_paths):
                    f.write(f"{py_file_path}\n")
            else:
                f.write("没有找到任何核心.py文件。\n")
            f.write("\n")

            # === 3. 获取Python文件代码内容 ===
            f.write("=== 核心Python文件代码内容: ===\n")
            f.write("\n")

            if py_files_paths:
                for py_file_path in sorted(py_files_paths):
                    f.write(f"\n--- 文件: {py_file_path} ---\n")
                    f.write("\n")
                    try:
                        with open(py_file_path, 'r', encoding='utf-8') as py_f:
                            f.write(py_f.read())
                        f.write("\n")
                        f.write("----------------------------------------------------\n")
                    except Exception as e:
                        f.write(f"无法读取文件内容: {e}\n")
                        f.write("----------------------------------------------------\n")
            else:
                f.write("没有可读取内容的.py文件。\n")

            f.write("\n")
            f.write("====================================================\n")
            f.write("  [AI 分析数据结束]\n")
            f.write("====================================================\n")

        print(f"报告已成功生成并保存到: {output_filename}")
        print("操作完成。")

    except Exception as e:
        print(f"发生了一个错误: {e}")

if __name__ == "__main__":
    # 获取脚本所在的目录作为分析的根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 允许用户通过命令行参数指定路径，否则使用脚本所在目录
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        # 对用户输入的路径进行规范化处理，以应对Windows和WSL路径的混合情况
        # os.path.abspath 可以处理相对路径，并将其转换为绝对路径
        # os.path.normpath 可以规范化路径，处理'..' '.' 等
        # 注意：对于 \\wsl.localhost\Ubuntu-22.04\home\zhz 这样的路径，os模块通常能正确处理
        processed_path = os.path.normpath(os.path.abspath(input_path))
    else:
        processed_path = script_dir

    generate_project_info(processed_path)

