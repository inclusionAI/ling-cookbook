#!/usr/bin/env python3
"""
scripts/compile_notebooks.py
============================
单向编译工具：将 src/ 目录下的 MyST Markdown (*.md) 源文件编译为根目录下的 Jupyter Notebook (*.ipynb) 交付物。

用法:
    python3 scripts/compile_notebooks.py             # 全量编译 src/ 下所有 .md 到对应的 .ipynb
    python3 scripts/compile_notebooks.py --check     # 检查编译产物是否与源码严格一致 (CI 校验)
    python3 scripts/compile_notebooks.py --staged    # 仅编译暂存区有变动的 .md 并自动 git add 对应 .ipynb
    python3 scripts/compile_notebooks.py --export    # 将现有 .ipynb 反向导出回 src/ (首次或灾难恢复)
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

# 确保在受限沙箱或无权限环境中也能安全运行 jupytext
os.environ.setdefault("JUPYTER_DATA_DIR", "/tmp/jupyter")
os.environ.setdefault("JUPYTER_RUNTIME_DIR", "/tmp/jupyter")
os.environ.setdefault("JUPYTER_CONFIG_DIR", "/tmp/jupyter")
Path("/tmp/jupyter").mkdir(parents=True, exist_ok=True)

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

def find_md_files():
    return sorted(list(SRC_DIR.glob("**/*.md")))

def get_target_ipynb_path(md_path: Path) -> Path:
    rel_path = md_path.relative_to(SRC_DIR)
    return ROOT_DIR / rel_path.with_suffix(".ipynb")

def get_source_md_path(ipynb_path: Path) -> Path:
    rel_path = ipynb_path.relative_to(ROOT_DIR)
    return SRC_DIR / rel_path.with_suffix(".md")

def compile_single(md_path: Path, verbose: bool = True) -> bool:
    target_ipynb = get_target_ipynb_path(md_path)
    target_ipynb.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Jupytext 转为 ipynb (生成的 Notebook 天然无执行输出)
    cmd = ["jupytext", str(md_path), "--to", "ipynb", "-o", str(target_ipynb)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"❌ 编译失败: {md_path}\n{res.stderr}", file=sys.stderr)
        return False
    
    if verbose:
        print(f"✅ 编译成功: {md_path.relative_to(ROOT_DIR)} -> {target_ipynb.relative_to(ROOT_DIR)}")
    return True

def compile_all(verbose: bool = True) -> bool:
    md_files = find_md_files()
    if not md_files:
        print("未在 src/ 目录下找到任何 .md 文件。")
        return True
    
    success = True
    for md_file in md_files:
        if not compile_single(md_file, verbose=verbose):
            success = False
    return success

def export_all() -> bool:
    guide_dir = ROOT_DIR / "guide"
    ipynb_files = sorted(list(guide_dir.glob("**/*.ipynb")))
    for ipynb in ipynb_files:
        target_md = get_source_md_path(ipynb)
        target_md.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["jupytext", str(ipynb), "--to", "md:myst", "-o", str(target_md)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"❌ 导出失败: {ipynb}\n{res.stderr}", file=sys.stderr)
            return False
        print(f"📥 导出完成: {ipynb.relative_to(ROOT_DIR)} -> {target_md.relative_to(ROOT_DIR)}")
    return True

def check_staged_and_sync() -> int:
    """Pre-commit 专用的同步与拦截逻辑"""
    # 获取暂存区文件列表
    diff_res = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        capture_output=True,
        text=True
    )
    if diff_res.returncode != 0:
        return 0
    
    staged_files = [line.strip() for line in diff_res.stdout.splitlines() if line.strip()]
    staged_md = [f for f in staged_files if f.startswith("src/") and f.endswith(".md")]
    staged_ipynb = [f for f in staged_files if f.endswith(".ipynb")]
    
    # 检查是否有直接修改 .ipynb 而未修改对应 src/*.md 的非法操作
    illegal_direct_edits = []
    for ipynb in staged_ipynb:
        expected_md = f"src/{ipynb[:-6]}.md"
        if expected_md not in staged_md:
            illegal_direct_edits.append((ipynb, expected_md))
            
    if illegal_direct_edits:
        print("=" * 70, file=sys.stderr)
        print("❌ [Pre-commit 拦截] 检测到直接修改了发布目录下的 .ipynb 产物！", file=sys.stderr)
        print("本项目采用单向编译架构（src/ -> 根目录），禁止直接编辑 .ipynb 文件。", file=sys.stderr)
        print("请修改对应的 Markdown 源文件：", file=sys.stderr)
        for ipynb, expected_md in illegal_direct_edits:
            print(f"  - {ipynb}  -->  请修改源码: {expected_md}", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        return 1
    
    # 对有变动的 .md 文件执行单向编译并 git add 产物
    if staged_md:
        print(f"🔄 检测到 {len(staged_md)} 个源码 .md 变动，开始单向编译至 .ipynb...")
        for md_rel in staged_md:
            md_path = ROOT_DIR / md_rel
            if md_path.exists():
                if compile_single(md_path, verbose=True):
                    target_ipynb = get_target_ipynb_path(md_path)
                    subprocess.run(["git", "add", str(target_ipynb)])
    return 0

def check_ci() -> int:
    """CI 校验模式：确保根目录下所有的 .ipynb 与 src/ 下的 .md 完全同步且 git 无 diff"""
    compile_all(verbose=False)
    diff_res = subprocess.run(["git", "diff", "--exit-code", "*.ipynb"], capture_output=True, text=True)
    if diff_res.returncode != 0:
        print("❌ [CI 失败] 检测到仓库中的 .ipynb 产物与 src/*.md 源文件存在失步差异！", file=sys.stderr)
        print("请在本地运行 `python3 scripts/compile_notebooks.py` 重新编译并提交产物。", file=sys.stderr)
        return 1
    print("✅ [CI 校验通过] 所有的 .ipynb 产物与 src/*.md 源码严格对齐。")
    return 0

def main():
    parser = argparse.ArgumentParser(description="Ling Cookbook Notebook 编译与同步工具")
    parser.add_argument("--check", action="store_true", help="CI 校验模式：检查产物与源码是否一致")
    parser.add_argument("--staged", action="store_true", help="Pre-commit 模式：仅编译暂存区变动的源码并拦截非法修改")
    parser.add_argument("--export", action="store_true", help="导出模式：从现有的 .ipynb 反向生成 src/*.md")
    args = parser.parse_args()

    if args.check:
        sys.exit(check_ci())
    elif args.staged:
        sys.exit(check_staged_and_sync())
    elif args.export:
        success = export_all()
        sys.exit(0 if success else 1)
    else:
        success = compile_all()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
