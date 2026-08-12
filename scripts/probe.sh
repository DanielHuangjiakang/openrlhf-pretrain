#!/usr/bin/env bash
# 环境探测 —— 只读，约 30 秒。把全部输出复制回来即可。
#
# 唯一的写操作是磁盘测速：在候选目录下写一个 256MB 临时文件，读完立即删除。
# 除此之外不安装、不下载、不修改任何东西。
echo "================ ENV PROBE $(date -u +%FT%TZ) ================"

echo "--- host ---"
hostname; uname -srm
(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME") || echo "os-release: n/a"
ldd --version 2>/dev/null | head -1

echo "--- gpu ---"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
  echo "CUDA (driver runtime): $(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -1)"
else
  echo "nvidia-smi NOT FOUND"
fi
echo "--- gpu topology (NCCL 走什么链路) ---"
nvidia-smi topo -m 2>/dev/null || echo "n/a"

echo "--- cpu / ram ---"
echo "cores: $(nproc 2>/dev/null)"
free -g 2>/dev/null | awk '/Mem:/{print "ram: "$2" GB total, "$7" GB available"}'

echo "--- disk (需要约 1TB) ---"
for p in / /home /data /mnt /scratch /workspace /raid "$HOME"; do
  [ -d "$p" ] && df -h "$p" 2>/dev/null | awk -v P="$p" 'NR==2{printf "  %-14s %6s free of %-6s  (%s)\n", P, $4, $2, $6}'
done | sort -u

echo "--- disk write speed (256MB, 写完即删) ---"
for p in /data /scratch /workspace "$HOME"; do
  if [ -d "$p" ] && [ -w "$p" ]; then
    t="$p/.probe_speed_$$"
    s=$(dd if=/dev/zero of="$t" bs=1M count=256 conv=fdatasync 2>&1 | tail -1)
    rm -f "$t"; echo "  $p -> $s"
  fi
done

echo "--- python / 包管理 ---"
for c in python3 python3.10 python3.11 python3.12 pip3 uv conda mamba docker git; do
  printf "  %-10s %s\n" "$c" "$(command -v $c 2>/dev/null || echo '-')"
done
python3 -c "import sys; print('  default python:', sys.version.split()[0])" 2>/dev/null
nvcc --version 2>/dev/null | tail -1 || echo "  nvcc: not on PATH"

echo "--- 调度器 / 容器 ---"
command -v sbatch >/dev/null && echo "  SLURM: YES (sbatch present)" || echo "  SLURM: no"
command -v qsub  >/dev/null && echo "  PBS: YES" || echo "  PBS: no"
[ -f /.dockerenv ] && echo "  running inside a container: YES" || echo "  container: no"

echo "--- sudo ---"
sudo -n true 2>/dev/null && echo "  passwordless sudo: YES" || echo "  sudo needs a password (or unavailable)"

echo "--- 外网可达性 (计算节点上跑, 不要在登录节点跑) ---"
for u in https://pypi.org/simple/ https://huggingface.co https://github.com; do
  code=$(curl -s -o /dev/null -m 12 -w '%{http_code}' "$u" 2>/dev/null)
  printf "  %-32s %s\n" "$u" "${code:-FAIL}"
done
echo -n "  HF 下载实测: "
curl -s -m 25 -o /dev/null -w '%{speed_download} B/s\n' \
  https://huggingface.co/api/models/allenai/OLMo-1B 2>/dev/null || echo FAIL

echo "--- InfiniBand ---"
command -v ibstat >/dev/null && ibstat -l 2>/dev/null | head -4 || echo "  ibstat: not found (走 NVLink/PCIe 即可)"

echo "================ END ================"
