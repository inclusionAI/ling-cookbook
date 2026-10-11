---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.5
kernelspec:
  name: python3
  language: python
  display_name: Python 3 (ipykernel)
---
##### Copyright 2026 Ant Group.

Licensed under the MIT License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://opensource.org/licenses/MIT

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

+++

# Ling-3.0-flash on DGX Spark (llama.cpp Q4_K_M GGUF) 部署指南

<table align="left">
  <td>
    <a target="_blank" href="https://www.ant-ling.com/"><img src="https://img.shields.io/badge/Official_Website-ant--ling.com-6366f1?style=flat-square" alt="Website" /></a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/inclusionAI/ling-cookbook"><img src="https://img.shields.io/badge/GitHub-Ling_Cookbook-181717?style=flat-square&logo=github" alt="GitHub" /></a>
  </td>
  <td>
    <a target="_blank" href="https://huggingface.co/inclusionAI"><img src="https://img.shields.io/badge/Hugging_Face-inclusionAI-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="Hugging Face" /></a>
  </td>
  <td>
    <a target="_blank" href="https://www.modelscope.cn/organization/inclusionAI"><img src="https://img.shields.io/badge/ModelScope-inclusionAI-624AFF?style=flat-square" alt="ModelScope" /></a>
  </td>
  <td>
    <a target="_blank" href="https://x.com/AntLingAGI"><img src="https://img.shields.io/badge/X-@AntLingAGI-000000?style=flat-square&logo=x" alt="X" /></a>
  </td>
  <td>
    <a target="_blank" href="https://discord.com/invite/GNaQc8WC5T"><img src="https://img.shields.io/badge/Discord-Ling_Community-5865F2?style=flat-square&logo=discord&logoColor=white" alt="Discord" /></a>
  </td>
</table>
<br><br>

+++

Ling-3.0-flash 是总参数量 124B、单 Token 激活 5.1B 的 MoE 大语言模型。

本 Notebook 将采用 `llama.cpp`，在 `NVIDIA DGX Spark` 设备上对其进行 4-bit 量化（Q4_K_M）版本的部署。量化后，模型权重占用压缩至 ~60.5 GB，可在单台 DGX Spark 上提供快速稳定的并发服务。

> [!TIP]
> **环境准备建议**：
> - 推荐使用 **Python 3.11 / 3.12** 环境；
> - 推荐使用 **uv** 创建独立的 Python 虚拟环境，以确保依赖隔离与工具链兼容性。

+++

### 步骤 1: 准备 Python 虚拟环境 (uv) 与克隆 llama.cpp 仓库

推荐使用 `uv` 创建独立虚拟环境并安装 OpenAI Python 包以用于测试。`llama.cpp` 官方主干已合并对 Ling-3.0 架构的支持，直接克隆主干源码即可：

```{code-cell}
!pip install -U uv
!uv venv --python 3.11 .venv
!source .venv/bin/activate && uv pip install --upgrade 'openai>=1.52.0,<2.0.0'
!git clone https://github.com/ggerganov/llama.cpp.git
```

典型输出：

```text
Using CPython 3.11 interpreter at: /usr/bin/python3.11
Creating virtualenv at: .venv
Cloning into 'llama.cpp'...
remote: Enumerating objects: 45210, done.
remote: Counting objects: 100% (210/210), done.
```

+++

### 步骤 2: 构建 llama.cpp

在 DGX Spark 上启用 CUDA 加速编译 `llama.cpp`：

```{code-cell}
!cd llama.cpp && cmake -B build -DGGML_CUDA=ON . && cmake --build build --parallel 8
```

典型运行输出：
```text
-- The CXX compiler identification is GNU 11.4.0
-- The CUDA compiler identification is NVIDIA 12.8.55
-- Detecting CUDA compiler ABI info - done
-- Building with CUDA architecture: native
[ 50%] Building CXX object CMakeFiles/llama.dir/src/llama.cpp.o
[100%] Built target llama-server
[100%] Built target llama-quantize
```

+++

### 步骤 3: 下载 Ling-3.0-flash 模型权重文件

我们需要先获取模型的原始权重文件。如果你在中国，推荐使用 [ModelScope CLI](https://github.com/modelscope/modelscope/blob/master/README_zh.md) 下载；或者，你也可以使用 [Hugging Face CLI](https://huggingface.co/docs/hub/agents-cli) 下载权重。

该模型在这两个模型托管平台的主页是 [Ling-3.0-flash on ModelScope](https://modelscope.cn/models/inclusionAI/Ling-3.0-flash) 和 [Ling-3.0-flash on Hugging Face](https://huggingface.co/inclusionAI/Ling-3.0-flash) 。

以下以 ModelScope 为例：

```{code-cell}
!source .venv/bin/activate && uv pip install -U modelscope
!source .venv/bin/activate && uv run modelscope download --model inclusionAI/Ling-3.0-flash --local-dir ~/models/Ling-3.0-flash
```

典型运行输出：
```text
Downloading [config.json, model.safetensors.index.json, ...]
Downloading shard 1/24: 100%|██████████| 4.98G/4.98G [00:15<00:00, 332MB/s]
...
Downloading shard 24/24: 100%|██████████| 3.12G/3.12G [00:09<00:00, 346MB/s]
Successfully downloaded Ling-3.0-flash to ~/models/Ling-3.0-flash
```

+++

### 步骤 4: 将模型转换为全精度 GGUF 格式

`llama.cpp` 支持使用 GGUF 格式模型。我们安装转换依赖，并使用 `convert_hf_to_gguf.py` 将 Safetensors 权重转换为全精度 BF16 的 GGUF 格式：

```{code-cell}
!source .venv/bin/activate && uv pip install -r ./llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
!source .venv/bin/activate && python3 llama.cpp/convert_hf_to_gguf.py ~/models/Ling-3.0-flash \
  --outfile ~/models/Ling-3.0-flash-bf16.gguf \
  --outtype bf16 --model-name Ling-3.0-flash
```

典型运行输出：
```text
INFO:hf-to-gguf:Loading model: Ling-3.0-flash
INFO:hf-to-gguf:Set model parameters
INFO:hf-to-gguf:Writing tensors to /home/squall/models/Ling-3.0-flash-bf16.gguf
INFO:hf-to-gguf:Done writing 124B model tensors (bf16).
```

+++

### 步骤 5: 量化模型到 Q4_K_M GGUF 格式

该模型的全精度版本无法在单台 DGX Spark 上运行，我们使用编译生成的 `llama-quantize` 工具将 BF16 GGUF 压缩为 4-bit (Q4_K_M) 格式，模型体积从 ~248GB 降至 ~60.5GB，适配 DGX Spark 单机运行。

```{code-cell}
!./llama.cpp/build/bin/llama-quantize ~/models/Ling-3.0-flash-bf16.gguf ~/models/Ling-3.0-flash-Q4_K_M.gguf Q4_K_M
```

典型运行输出：
```text
main: build = 26608
main: quantizing '/home/squall/models/Ling-3.0-flash-bf16.gguf' to '/home/squall/models/Ling-3.0-flash-Q4_K_M.gguf' as Q4_K_M
[   1/ 780]              blk.0.attn_q.weight - [ 2560,  2560,     1], type =  f16, size =    12.50 MB -> Q4_K, size =     3.44 MB
...
[ 780/ 780]                     output.weight - [ 2560, 157184,     1], type =  f16, size =   767.50 MB -> Q6_K, size =   300.00 MB
main: model size  = 238410.20 MB -> 60512.44 MB
main: quantize time = 184512.20 ms
```

+++

### 步骤 6: 用 llama-server 启动模型推理服务

启动 `llama-server`，提供 OpenAI 格式兼容的 HTTP 接口。部分部署参数说明：

- `-ngl all`：将所有网络层部署至 GPU 运算
- `-fa on`：开启 FlashAttention 加速
- `-c 262144`：启用 256K 上下文窗口
- `-np 4`：支持 4 路并发处理

如未指定，llama-server 默认使用 8080 端口。我们这里选择了端口 `9102`。

⚠️ 特别注意：

`llama-server` 以前台常驻模式运行。如果你直接在 Notebook 中运行下方单元格，Jupyter 将阻塞而无法执行其他单元格的代码。建议你在单独的终端会话中执行启动命令。启动成功后即可使用 Jupyter 进行后续验证。

```{code-cell}
!./llama.cpp/build/bin/llama-server \
  -m ~/models/Ling-3.0-flash-Q4_K_M.gguf \
  --alias Ling-3.0-flash-Q4_K_M \
  -ngl all -fa on -c 262144 -cb -np 4 \
  --host 0.0.0.0 --port 9102 --api-key sk-ling-cookbook-test
```

服务端就绪时的典型日志：
```text
llama_server: HTTP server listening on 0.0.0.0:9102
llama_server: model loaded successfully, n_ctx = 262144, offload = 100% (GPU)
```

+++

### 步骤 7: 验证部署成功并使用模型服务

服务启动后，即可在各种 LLM 客户端中使用。下面提供的例子包含：

1. 健康检查 - 确认模型加载状态与端点连通性。
2. 流式 Reasoning 验证 - 测试 `<think>` 正常生成，同时测算 TTFT 与 TPS 性能指标。
3. Tool Calling 验证 - 测试 Function Calling 是否可正常使用。

+++

#### 步骤 7.1: 连通性与模型健康检查

请求 `GET /v1/models` 查看当前运行的模型信息：

```{code-cell}
import urllib.request

url = "http://localhost:9102/v1/models"
try:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        status_code = response.getcode()
        body = response.read().decode('utf-8')
        print(f"Health check status: {status_code}")
        print(f"Models response: {body}")
except Exception as e:
    print(f"Health check error: {e}")
```

典型输出：

```
Health check status: 200
Models response: {"models":[{"name":"Ling-3.0-flash-Q4_K_M","model":"Ling-3.0-flash-Q4_K_M","modified_at":"","size":"","digest":"","type":"model","description":"","tags":[""],"capabilities":["completion"],"parameters":"","details":{"parent_model":"","format":"gguf","family":"","families":[""],"parameter_size":"","quantization_level":""}}],"object":"list","data":[{"id":"Ling-3.0-flash-Q4_K_M","aliases":["Ling-3.0-flash-Q4_K_M"],"tags":[],"object":"model","created":1786790315,"owned_by":"llamacpp","meta":{"vocab_type":2,"n_vocab":157184,"n_ctx":65536,"n_ctx_train":262144,"n_embd":2560,"n_params":127486405600,"size":77003601792,"ftype":"Q4_K - Medium"}}]}
```

+++

#### 步骤 7.2: 流式推理、Reasoning 和速度测量

使用 OpenAI Python SDK 调用该接口，通过 `enable_thinking: True` 提取 `<think>` 思考链内容，并统计 TTFT 与 TPS：

```{code-cell}
import time
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:9102/v1",
    api_key="sk-ling-cookbook-test"
)

def verify_streaming_and_thinking():
    prompt = "请推导公式 17 × 23 并详细给出计算步骤。"
    print(f"Sending prompt: '{prompt}' to model 'Ling-3.0-flash-Q4_K_M'...")

    start_time = time.time()
    first_token_time = None
    total_tokens = 0
    reasoning_text = ""
    content_text = ""

    try:
        response = client.chat.completions.create(
            model="Ling-3.0-flash-Q4_K_M",
            messages=[
                {"role": "user", "content": prompt}
            ],
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
            temperature=0.6,
            top_p=0.95,
            stream=True
        )

        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            now = time.time()
            if first_token_time is None:
                first_token_time = now

            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                reasoning_text += delta.reasoning_content
                total_tokens += 1
            elif delta.content:
                content_text += delta.content
                total_tokens += 1

        end_time = time.time()
        ttft = (first_token_time - start_time) * 1000.0 if first_token_time else 0.0
        decode_duration = end_time - first_token_time if first_token_time else 0.001
        tps = total_tokens / decode_duration

        print("\n=== Latency & Throughput Metrics ===")
        print(f"TTFT (Time to First Token): {ttft:.2f} ms")
        print(f"Decode TPS (Tokens/s): {tps:.2f} t/s")
        print(f"Total Generated Tokens: {total_tokens}")
        print(f"Total Duration: {end_time - start_time:.2f} s")

        print("\n=== Extracted Reasoning Chain (<think>) ===")
        print(reasoning_text if reasoning_text else "[Note: Reasoning merged in main content or parsed separately]")

        print("\n=== Final Response Content ===")
        print(content_text)

    except Exception as e:
        print(f"Streaming verification failed: {e}")

if __name__ == "__main__":
    verify_streaming_and_thinking()
```

典型测试结果：

```
Sending prompt: '请推导公式 17 × 23 并详细给出计算步骤。' to model 'Ling-3.0-flash-Q4_K_M'...

=== Latency & Throughput Metrics ===
TTFT (Time to First Token): 667.49 ms
Decode TPS (Tokens/s): 46.19 t/s
Total Generated Tokens: 614
Total Duration: 13.96 s
...
```

+++

#### 步骤 7.3: 测试 Function Calling 和结构化输出

传入标准 Function Calling Schema（以天气查询为例），验证模型对结构化工具调用的支持：

```{code-cell}
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:9102/v1",
    api_key="sk-ling-cookbook-test"
)

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的实时天气预报与气温信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如：杭州、北京、上海"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度单位"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

def verify_tool_calling():
    print(f"Testing Function Calling / Tool Use with model 'Ling-3.0-flash-Q4_K_M'...")
    try:
        response = client.chat.completions.create(
            model="Ling-3.0-flash-Q4_K_M",
            messages=[
                {"role": "user", "content": "请帮我查一下杭州今天的天气如何？"}
            ],
            tools=tools_schema,
            tool_choice="auto"
        )

        message = response.choices[0].message
        if message.tool_calls:
            print("\n=== Function Call Output Detected ===")
            for tool_call in message.tool_calls:
                print(f"Tool Call ID: {tool_call.id}")
                print(f"Function Name: {tool_call.function.name}")
                print(f"Arguments JSON: {tool_call.function.arguments}")
        else:
            print("\n=== Direct Response (No Tool Call Triggered) ===")
            print(message.content)

    except Exception as e:
        print(f"Tool calling verification failed: {e}")

if __name__ == "__main__":
    verify_tool_calling()
```

典型测试结果：

```
Testing Function Calling / Tool Use with model 'Ling-3.0-flash-Q4_K_M'...

=== Function Call Output Detected ===
Tool Call ID: gpGiPyVIVoeg7Y4ZWyqKgjfFacOVCAeC
Function Name: get_weather
Arguments JSON: {"city":"杭州"}
```

+++

### 步骤 8: 常见问题与故障排查

1. **GGUF 转换依赖安装报错**：
   - 现象：运行 `convert_hf_to_gguf.py` 时报 `ModuleNotFoundError`。
   - 解决：确保在虚拟环境中执行 `uv pip install -r ./llama.cpp/requirements/requirements-convert_hf_to_gguf.txt` 安装转换所需的全部依赖库（如 `torch`, `sentencepiece` 等）。

2. **多并发下上下文显存分配**：
   - 现象：启动时指定超大上下文（如 `-c 262144`）与较高并发数（如 `-np 4`）时显存溢出。
   - 解决：可适度减小并发槽位 `-np 2` 或降低上下文长度 `-c 131072`。

3. **端口冲突 (Port 9102 occupied)**：
   - 现象：`llama-server` 启动报错端口已被占用。
   - 解决：通过 `lsof -i :9102` 查找并结束进程，或在启动命令中通过 `--port` 修改端口。

