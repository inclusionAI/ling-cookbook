#!/usr/bin/env python3
"""Illustrative inference: Ling-3.0-tiny + the released LoRA adapter.
Requires an actual GPU environment and extracted adapter directory.
Produces a tool-call JSON OBJECT only; never executes tools.

Example:
  tar -xzf adapter.tar.gz
  python inference-example.py --adapter ./adapter --question '列出 docs/ 的文件'
"""
import argparse
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL_ID = "inclusionAI/Ling-3.0-tiny"
SYSTEM_PROMPT = '''你是一个严谨的端侧只读 Agent。你只能通过生成合法 JSON 来调用以下 4+1 个受限工具：
1. `list_files(directory: str)`: 查看指定只读工作区目录下的文件清单。
2. `read_file(path: str, max_lines: int = 100)`: 读取指定代码、配置或文档文件内容（最多 100 行）。
3. `search_files(pattern: str, directory: str)`: 根据文件名模式或通配符检索文件路径。
4. `text_retrieval(query: str, top_k: int = 5)`: 在本地文档与知识库中进行语义及关键词检索。
5. `reject_none(reason: str)`: 安全防御拦截。当且仅当用户请求涉及写入、删除、联网请求、执行任意 shell 命令或试图越界访问工作区之外的敏感路径时，必须拒绝并说明理由。

严格要求：
- 只能返回合法 JSON 对象，格式：{"tool": "<工具名>", "arguments": {<参数映射>}}
- 严禁输出任何多余的开场白、解释说明或 markdown 标记之外的内容。'''


def first_tool_call(text: str):
    decoder=json.JSONDecoder()
    for offset, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, end=decoder.raw_decode(text[offset:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj,dict) and isinstance(obj.get("tool"),str) and isinstance(obj.get("arguments"),dict):
            return obj
        return None
    return None


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--adapter",type=Path,required=True,help="Extracted directory containing adapter_config.json")
    parser.add_argument("--question",default="列出 docs/ 文件夹中的文件")
    args=parser.parse_args()
    if not (args.adapter/"adapter_config.json").exists():
        raise SystemExit("Adapter directory missing adapter_config.json; extract the archived adapter first")
    if not torch.cuda.is_available():
        raise SystemExit("This example requires a CUDA GPU; run in a Colab GPU session")
    tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,trust_remote_code=True)
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    base=AutoModelForCausalLM.from_pretrained(MODEL_ID,quantization_config=bnb,device_map="auto",trust_remote_code=True)
    model=PeftModel.from_pretrained(base,str(args.adapter))
    model.eval()
    # Reproduce the serialized prompt convention in the recorded training script.
    prompt=f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{args.question}<|im_end|>\n<|im_start|>assistant\n"
    toks=tokenizer(prompt,return_tensors="pt")
    inputs={k:v.to(model.device) for k,v in toks.items() if k in ("input_ids","attention_mask")}
    with torch.inference_mode():
        out=model.generate(**inputs,max_new_tokens=128,do_sample=False,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
    raw=tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],skip_special_tokens=False)
    tool=first_tool_call(raw)
    if tool is None:
        print("Could not parse a tool call. Raw output:\n"+raw)
        raise SystemExit(2)
    print(json.dumps(tool,ensure_ascii=False,indent=2))
    print("\nNote: the example parses the first tool-call object; raw trailing dialogue is not claimed to be strict JSON-only output.")


if __name__=="__main__":main()
