"""Score JEV decision checkpoints on CPU, for machines where the GPU may not be used.

AReno has CUDA and MLX backends only, so this file recomputes exactly what its CUDA
`DecisionScorer` computes, with the Hugging Face model code on CPU:

  prompt     "<|oj:state|>\n{state}\n<|oj:question|>\nnoul: {instructions}\nyes := ...\nno := ...\n"
  candidates "no\n<|oj:answer|>" and "yes\n<|oj:answer|>", each appended to the prompt
  pooling    normalized final hidden state of the last token (`last_valid_token`)
  head       Linear(1536, 1536) -> GELU -> Linear(1536, 1), FP32
  answer     softmax over the two candidate logits / temperature; P(unsafe) = P("yes")

The LoRA adapter is merged into the base weights. The Triton kernels the model code imports from
flash-linear-attention are replaced by the library's own PyTorch reference implementations.
Check it against GPU predictions first:

    CUDA_VISIBLE_DEVICES= python cpu_scorer.py --arm a7j_s110 --checkpoint ../adapters/a7j_s110 \
        --model ../models/Ling-3.0-tiny --bench-root ../bench --slice dev --limit 20 \
        --compare-dir ../runs/preds_dev --out-dir /tmp/cpu_check
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch import nn

from benches import load_all, render_state
from jevdata import QUESTION

STATE_MARK, QUESTION_MARK, ANSWER_MARK = "<|oj:state|>", "<|oj:question|>", "<|oj:answer|>"


def question_prefix(state: str, question: dict) -> str:
    lines = [STATE_MARK, state, QUESTION_MARK, f"{question['type']}: {question['instructions']}"]
    criteria = question.get("criteria") or {}
    if "true" in criteria:
        lines.append(f"yes := {criteria['true']}")
    if "false" in criteria:
        lines.append(f"no := {criteria['false']}")
    return "\n".join(lines) + "\n"


def canonical_state(state: str) -> str:
    """AReno serializes JSON text states canonically; plain text passes through."""
    try:
        return json.dumps(json.loads(state), ensure_ascii=False, sort_keys=True)
    except ValueError:
        return state


# ---- CPU stand-ins for the Triton kernels -----------------------------------------------------


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.float()
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + eps)


def kda_gate(g, A_log, dt_bias, lower_bound):
    heads, dim = g.shape[-2], g.shape[-1]
    g = g.float()
    if dt_bias is not None:
        g = g + dt_bias.float().view(heads, dim)
    scale = A_log.float().view(heads, 1).exp()
    if lower_bound is not None:
        return lower_bound * torch.sigmoid(scale * g)
    return -scale * F.softplus(g)


def kda_cpu(q, k, v, g, beta, A_log=None, dt_bias=None, scale=None, initial_state=None, output_final_state=False,
            use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False, safe_gate=False, lower_bound=None,
            cu_seqlens=None, **_unused):
    from fla.ops.kda.naive import naive_recurrent_kda

    if cu_seqlens is not None:
        raise NotImplementedError("packed sequences are not used by this scorer")
    if use_qk_l2norm_in_kernel:
        q, k = l2norm(q), l2norm(k)
    if use_gate_in_kernel:
        g = kda_gate(g, A_log, dt_bias, lower_bound)
    out, state = naive_recurrent_kda(q.float(), k.float(), v.float(), g.float(), beta.float(), scale=scale,
                                     initial_state=initial_state, output_final_state=output_final_state)
    return out.to(v.dtype), state


def gated_rmsnorm_forward(self, x, g, residual=None, prenorm=False, residual_in_fp32=False):
    dtype = x.dtype
    x = x.float()
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    if getattr(self, "weight", None) is not None:
        y = y * self.weight.float()
    if getattr(self, "bias", None) is not None:
        y = y + self.bias.float()
    activation = getattr(self, "activation", "sigmoid")
    gate = g.float()
    y = y * (torch.sigmoid(gate) if activation == "sigmoid" else F.silu(gate))
    return y.to(dtype)


def short_conv_forward(self, x, residual=None, mask=None, cache=None, output_final_state=False, cu_seqlens=None, **_):
    """Depthwise causal conv. The cache holds the last kernel-1 inputs, so a suffix can continue a prefix."""
    if cu_seqlens is not None:
        raise NotImplementedError("packed sequences are not used by this scorer")
    weight = self.weight  # [channels, 1, kernel]
    width = weight.shape[-1]
    inputs = x.transpose(1, 2).float()
    history = cache.float() if cache is not None else inputs.new_zeros(*inputs.shape[:2], width - 1)
    padded = torch.cat([history, inputs], dim=-1)
    y = F.conv1d(padded, weight.float(), None if self.bias is None else self.bias.float(), groups=weight.shape[0])
    activation = getattr(self, "activation", None)
    if activation in ("silu", "swish"):
        y = F.silu(y)
    new_cache = padded[..., -(width - 1):].contiguous() if output_final_state else None
    return y.transpose(1, 2).to(x.dtype), new_cache


def patch_model_code(model) -> None:
    import fla.modules

    module = sys.modules[model.__class__.__module__]
    module.chunk_kda = kda_cpu
    module.fused_recurrent_kda = kda_cpu
    fla.modules.FusedRMSNormGated.forward = gated_rmsnorm_forward
    fla.modules.ShortConvolution.forward = short_conv_forward


# ---- checkpoint loading -------------------------------------------------------------------------


def merge_lora(model, checkpoint: str) -> int:
    from safetensors.torch import load_file

    with open(os.path.join(checkpoint, "adapter_config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    scaling = float(config["lora_alpha"]) / float(config["r"])
    tensors = load_file(os.path.join(checkpoint, "adapter_model.safetensors"))
    merged = 0
    for name, lora_a in tensors.items():
        if not name.endswith(".lora_A.weight"):
            continue
        path = name[len("base_model.model."):-len(".lora_A.weight")]
        lora_b = tensors[name.replace(".lora_A.", ".lora_B.")]
        layer = model.get_submodule(path)
        delta = (lora_b.float() @ lora_a.float()) * scaling
        with torch.no_grad():
            layer.weight.copy_((layer.weight.float() + delta).to(layer.weight.dtype))
        merged += 1
    return merged


def lora_side_forward(self, x):
    """Base projection plus the LoRA path computed separately, as AReno's training forward does."""
    out = F.linear(x, self.weight, self.bias)
    return out + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scale


def attach_lora(model, checkpoint: str, dtype: torch.dtype) -> int:
    import types

    from safetensors.torch import load_file

    with open(os.path.join(checkpoint, "adapter_config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    scaling = float(config["lora_alpha"]) / float(config["r"])
    tensors = load_file(os.path.join(checkpoint, "adapter_model.safetensors"))
    attached = 0
    for name, lora_a in tensors.items():
        if not name.endswith(".lora_A.weight"):
            continue
        layer = model.get_submodule(name[len("base_model.model."):-len(".lora_A.weight")])
        layer.lora_A = lora_a.to(dtype)
        layer.lora_B = tensors[name.replace(".lora_A.", ".lora_B.")].to(dtype)
        layer.lora_scale = scaling
        layer.forward = types.MethodType(lora_side_forward, layer)
        attached += 1
    return attached


def load_head(checkpoint: str, hidden: int) -> nn.Module:
    from safetensors.torch import load_file

    head = nn.Module()
    head.layers = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
    head.load_state_dict(load_file(os.path.join(checkpoint, "score_head.safetensors")))
    return head.float().eval()


class CpuScorer:
    def __init__(self, model_path: str, checkpoint: str, dtype: torch.dtype, lora_mode: str = "merge"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        with open(os.path.join(checkpoint, "jev_config.json"), encoding="utf-8") as handle:
            config = json.load(handle)
        self.temperature = float(config.get("temperature", 1.0))
        self.max_length = int(config.get("max_seq_len", 1024))
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=dtype)
        patch_model_code(model)
        self.merged = merge_lora(model, checkpoint) if lora_mode == "merge" else attach_lora(model, checkpoint, dtype)
        self.backbone = model.model.eval()
        self.head = load_head(checkpoint, model.config.hidden_size)

    def encode_parts(self, state: str, question: dict) -> tuple[list[int], list[list[int]]]:
        candidates = ["no\n" + ANSWER_MARK, "yes\n" + ANSWER_MARK]
        suffixes = [self.tokenizer.encode(c, add_special_tokens=False) for c in candidates]
        budget = self.max_length - max(map(len, suffixes))
        prefix = self.tokenizer.encode(question_prefix(canonical_state(state), question), add_special_tokens=False)
        return prefix[-budget:], suffixes

    def encode(self, state: str, question: dict) -> list[list[int]]:
        prefix, suffixes = self.encode_parts(state, question)
        return [prefix + suffix for suffix in suffixes]

    def candidate_logits(self, state: str, question: dict, share_prefix: bool) -> list[float]:
        if not share_prefix:
            logits = []
            for tokens in self.encode(state, question):
                ids = torch.tensor([tokens], dtype=torch.long)
                hidden = self.backbone(input_ids=ids, use_cache=False).last_hidden_state[0, -1]
                logits.append(float(self.head.layers(hidden.float())))
            return logits
        # Run the shared prompt once, then each short candidate from a copy of its cache. Positions are
        # passed explicitly: the hybrid cache's default length lookup reads a linear-attention layer.
        import copy

        from transformers.cache_utils import DynamicCache

        prefix, suffixes = self.encode_parts(state, question)
        cache = DynamicCache()
        self.backbone(input_ids=torch.tensor([prefix], dtype=torch.long), past_key_values=cache, use_cache=True)
        logits = []
        for suffix in suffixes:
            positions = torch.arange(len(prefix), len(prefix) + len(suffix), dtype=torch.long)
            out = self.backbone(input_ids=torch.tensor([suffix], dtype=torch.long), past_key_values=copy.deepcopy(cache),
                                use_cache=True, cache_position=positions, position_ids=positions.unsqueeze(0))
            logits.append(float(self.head.layers(out.last_hidden_state[0, -1].float())))
        return logits

    @torch.inference_mode()
    def p_unsafe(self, state: str, question: dict = QUESTION, share_prefix: bool = False) -> float:
        logits = self.candidate_logits(state, question, share_prefix)
        scores = [value / self.temperature for value in logits]
        top = max(scores)
        weights = [math.exp(value - top) for value in scores]
        return weights[1] / sum(weights)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bench-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--benches", default="preex,asse-security,asse-safety,rjudge")
    parser.add_argument("--slice", choices=["dev", "test"], required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="records per bench, for checks")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="float32",
                        help="float32 keeps the merged LoRA deltas, which are often below a bfloat16 step of the base weight")
    parser.add_argument("--lora", choices=["merge", "side"], default="merge",
                        help="side keeps the LoRA path separate, the way the GPU training forward computes it")
    parser.add_argument("--share-prefix", action="store_true", help="run the prompt once for both candidates")
    parser.add_argument("--compare-dir", default="", help="GPU predictions to compare against")
    parser.add_argument("--only-missing", default="", help="score only records absent from <dir>/<arm>.<bench>.jsonl")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) not in ("", "-1"):
        raise SystemExit("set CUDA_VISIBLE_DEVICES= (empty): this scorer must not touch a GPU")
    torch.set_num_threads(args.threads)
    started = time.time()
    scorer = CpuScorer(args.model, args.checkpoint, getattr(torch, args.dtype), args.lora)
    print(f"{args.arm}: loaded in {time.time() - started:.0f}s, merged {scorer.merged} LoRA modules", flush=True)
    data = load_all(args.bench_root)
    os.makedirs(args.out_dir, exist_ok=True)
    for bench in filter(None, args.benches.split(",")):
        records = [r for r in data[bench] if r["split"] == args.slice]
        records = records[args.shard :: args.num_shards]
        if args.limit:
            records = records[: args.limit]
        if args.only_missing:
            path = os.path.join(args.only_missing, f"{args.arm}.{bench}.jsonl")
            done = set()
            if os.path.exists(path):
                with open(path, encoding="utf-8") as handle:
                    done = {row["id"] for row in map(json.loads, handle)}
            records = [r for r in records if r["id"] not in done]
        reference = {}
        if args.compare_dir:
            path = os.path.join(args.compare_dir, f"{args.arm}.{bench}.jsonl")
            if os.path.exists(path):
                with open(path, encoding="utf-8") as handle:
                    reference = {row["id"]: row["prob"] for row in map(json.loads, handle)}
        suffix = f".shard{args.shard}of{args.num_shards}" if args.num_shards > 1 else ""
        if args.only_missing:
            suffix += ".missing"
        out = os.path.join(args.out_dir, f"{args.arm}.{bench}{suffix}.jsonl")
        diffs, begun = [], time.time()
        with open(out, "w", encoding="utf-8") as handle:
            for record in records:
                prob = scorer.p_unsafe(render_state(record), share_prefix=args.share_prefix)
                handle.write(json.dumps({"id": record["id"], "prediction": int(prob >= 0.5), "prob": prob}) + "\n")
                handle.flush()
                if record["id"] in reference:
                    diffs.append(abs(prob - reference[record["id"]]))
        line = f"{args.arm} {bench}: {len(records)} in {time.time() - begun:.0f}s"
        if diffs:
            line += f" | vs GPU: mean |dp| {sum(diffs) / len(diffs):.4f}, max {max(diffs):.4f}"
        print(line, flush=True)


if __name__ == "__main__":
    main()
