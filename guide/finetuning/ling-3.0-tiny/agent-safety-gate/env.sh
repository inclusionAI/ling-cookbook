# Keeps HOME, caches and temp files inside the work directory. Set LING11_ROOT to move it.
export L=${LING11_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
export HOME=$L/home
export ARENO_MST_CHUNK_SIZE=0
export TMPDIR=$L/tmp TEMP=$L/tmp TMP=$L/tmp
export XDG_CACHE_HOME=$L/cache XDG_CONFIG_HOME=$L/cache/config XDG_DATA_HOME=$L/cache/data
export HF_HOME=$L/cache/hf MODELSCOPE_CACHE=$L/cache/ms PIP_CACHE_DIR=$L/cache/pip UV_CACHE_DIR=$L/cache/uv TORCH_HOME=$L/cache/torch
export UV_PYTHON_INSTALL_DIR=$L/python UV_TOOL_DIR=$L/tools/uv
export JUPYTER_DATA_DIR=$L/cache/jupyter MPLCONFIGDIR=$L/cache/mpl PYTHONPYCACHEPREFIX=$L/cache/pyc
export TRITON_CACHE_DIR=$L/cache/triton DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1
export PATH=$L/tools/bin:$PATH
mkdir -p "$L/home" "$L/tmp" "$L/cache"
if [ -f "$L/venv/bin/activate" ]; then . "$L/venv/bin/activate"; fi
true
