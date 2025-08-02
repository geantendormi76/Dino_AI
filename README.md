本项目采用基于`conda`的“双环境隔离”策略来管理复杂的GPU依赖。请遵循以下步骤来设置开发环境。

**先决条件:**

1.  已安装 [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/)。
2.  已安装最新的NVIDIA显卡驱动。

**步骤1：创建数据采集环境 (`dinoC`)**

打开终端（Anaconda Prompt或已配置好conda的PowerShell），进入项目根目录，然后运行以下命令：

```bash
conda env create -f environment-collect.yml
```

**步骤2：创建模型训练与执行环境 (`dinoT`)**

在同一个终端中，运行以下命令：

```bash
conda env create -f environment-train.yml
```

**步骤3：日常工作流**

*   **进行数据采集时:**
    ```bash
    conda activate dinoC
    python training_pipelines\3_policy_pipeline\1_collect_data.py
    ```

*   **进行模型训练、处理或最终运行时:**
    ```bash
    conda activate dinoT
    # 例如：
    python run_bot.py
    ```
---

