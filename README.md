# 可预测性探针（Predictability Probe）

**目的**：在训练自己的重构模型之前，先用两个只在真实视频上预训练过的现成模型，验证"真假视频在能否被预测/重构上是否存在稳定差异"。这一步**不训练任何模型**，假视频只用于评估。

| 探针 | 模型 | 测量层面 | 误差 |
|---|---|---|---|
| A | V-JEPA 2（torch.hub `facebookresearch/vjepa2`） | 特征层：运动、事件、语义 | 被遮挡 token 的特征 L1 |
| B | VideoMAE（HF `MCG-NJU/videomae-base`，带解码器的预训练权重） | 像素层：纹理、低层细节 | 被遮挡 tubelet 的像素 MSE |

## 目录

```
probe/video_io.py      ffprobe 元数据、片段规划、解码（PyAV 优先，没有则用 OpenCV）
probe/masks.py         遮挡方式：future / tube / block
probe/preprocess.py    缩放、裁剪（three = 左/中/右三块，覆盖整个 16:9 画面）、复杂度协变量
probe/backends.py      两个模型后端；保证上下文编码器看不到目标区域（不泄漏答案）
scripts/build_manifest.py   扫描数据文件夹，生成清单（标签、划分、分辨率/帧率/码率）
scripts/standardize.py      生成统一格式的副本（24fps、720p、H.264 CRF23），原始文件保留
scripts/selftest.py         在集群上必须先跑：检查模型加载、loss 是否一致、是否泄漏
scripts/run_probe.py        主程序：每条视频 × 片段 × 遮挡方式 × 随机种子，输出一行 JSON
scripts/analyze.py          汇总：按生成器算 AUC 和 95% 置信区间、双尾分数、复杂度校准、捷径检查、画图
slurm/*.sbatch              提交作业用的脚本
```

## 一、部署环境（在集群上逐项确认）

```bash
# 0) 本机 MSYS2 用 scp 把代码传到 slurm-datakom 的共享 /home/shuang。
#    从 datakom 用 Slurm 进入 hefest；不依赖直连 hefest 或 datakom 上的 git。
# 1) 从本机登录网关，申请一个交互式 GPU 会话
ssh slurm-datakom
mkdir -p ~/logs        # 只需做一次：Slurm 在作业启动前就要打开日志文件，目录不存在就拿不到日志
srun --partition=batch --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=0 --pty bash
SIF=/home/containers/pytorch-vision-h200.sif

# 2) 能否联网（下载模型权重需要）
curl -sI https://huggingface.co | head -1
curl -sI https://dl.fbaipublicfiles.com | head -1
curl -sI https://github.com | head -1

# 3) 容器里已经有什么
apptainer exec $SIF python3 -c "import torch,numpy,sklearn,cv2;print(torch.__version__,numpy.__version__,sklearn.__version__,cv2.__version__)"
apptainer exec $SIF which ffmpeg ffprobe
apptainer exec $SIF python3 -c "import transformers" ; apptainer exec $SIF python3 -c "import av"

# 4) 缺的包装进 overlay（注意固定 numpy 版本，避免破坏容器）
apptainer overlay create --size 10240 ~/overlay.img
apptainer exec --overlay ~/overlay.img $SIF pip install "numpy==1.26.4" "transformers<5" av matplotlib timm einops

# 5) ffmpeg/ffprobe 为独立可执行文件；见下方安装步骤，不需要 overlay
```

Hefest 的宿主机和容器都没有 `ffmpeg` / `ffprobe`。可以把第三方 Linux 静态版放进共享的 `/home/shuang/bin`，无须 root、overlay 或 GPU。在 `slurm-datakom` 上执行：

```bash
mkdir -p ~/bin ~/downloads/ffmpeg-static
cd ~/downloads/ffmpeg-static
curl -fL -o ffmpeg-release-amd64-static.tar.xz https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
curl -fL -o ffmpeg-release-amd64-static.tar.xz.md5 https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz.md5
md5sum -c ffmpeg-release-amd64-static.tar.xz.md5
# 只有校验显示 OK 后才继续
tar -xf ffmpeg-release-amd64-static.tar.xz
install -m 755 ffmpeg-*-amd64-static/ffmpeg ffmpeg-*-amd64-static/ffprobe ~/bin/
```

再从 datakom 申请一个 CPU 会话，到 Hefest 容器里验证，完成后退出：

```bash
srun --partition=batch --cpus-per-task=1 --mem=2G --time=0 --pty bash
apptainer exec --env PREPEND_PATH=/home/shuang/bin /home/containers/pytorch-vision-h200.sif ffprobe -version
exit
```

三个 Slurm 脚本已用 Apptainer 的 `PREPEND_PATH` 把 `~/bin` 加到容器的 PATH。若 datakom 无法下载，可在本机下载同一压缩包和校验文件后，用 MSYS2 `scp` 传到 datakom 的 `/home/shuang/downloads/ffmpeg-static/`，再从 `md5sum` 开始执行。

如果 hefest **不能联网**：在本机下载权重，用 `scp` 先传到 datakom 的共享 `/home/shuang/`，再在 Slurm 分配的 Hefest 会话中复制到 `/data/checkpoints/shuang/cache/`。V-JEPA 2 用的 torch.hub 需要源码仓库加权重文件；VideoMAE 用 HF 格式的目录，下载后通过 `--model-id /path/to/dir` 指定。

## 二、自检（必须全部 PASS 才能跑全量）

```bash
cd "/home/shuang/projects/Deepfake Reconstruction/predictability_probe"
export HF_HOME=/data/checkpoints/shuang/cache/hf TORCH_HOME=/data/checkpoints/shuang/cache/torch
SIF=/home/containers/pytorch-vision-h200.sif
OVERLAY_ARGS=()
[ -f ~/overlay.img ] && OVERLAY_ARGS=(--overlay ~/overlay.img:ro)
apptainer exec --nv "${OVERLAY_ARGS[@]}" --env PREPEND_PATH=/home/shuang/bin -B /data "$SIF" python3 scripts/selftest.py --backend videomae --video /data/datasets/shuang/magic/real/real_videos_mixkit_720p/134.mp4
apptainer exec --nv "${OVERLAY_ARGS[@]}" --env PREPEND_PATH=/home/shuang/bin -B /data "$SIF" python3 scripts/selftest.py --backend vjepa2   --video /data/datasets/shuang/magic/real/real_videos_mixkit_720p/134.mp4
```

检查项：
1. 模型能加载，输入尺寸正确；
2. VideoMAE：自己计算的逐 token 误差和 HF 内部的 loss 一致；
3. **泄漏检查**：把目标区域的像素换成噪声后，预测结果不应改变；
4. 把上下文换成噪声后，预测结果应该明显改变；
5. 单个片段的耗时和显存占用。

**V-JEPA 2 权重（已核实的上游问题，2026-11-06）**：上游 `main` 分支的 `src/hub/backbones.py` 把权重地址写成了 `http://localhost:8300`（官方地址被注释掉了），而且加载时用的是 `strict=False`。所以本代码**只用 hub 来构建网络结构（`pretrained=False`），权重自己下载**：默认从官方地址 `https://dl.fbaipublicfiles.com/vjepa2/vitl.pt`（约 5.1GB）下载，也可以用 `--weights /本地/vitl.pt` 指定本地文件。加载时会检查键名，**缺少权重就直接报错**，避免悄悄用随机初始化的预测器跑完全程。上下文编码器用 checkpoint 里的 `encoder`，目标编码器用 `target_encoder`。
建议第一次成功后把代码版本固定下来：`--hub-repo facebookresearch/vjepa2:<commit-sha>`。

V-JEPA 2 的 hub 接口是按 `encoder(x, masks=[idx])` 和 `predictor(ctx, ctx_idx, tgt_idx)` 写的。**如果 selftest 报接口不匹配，只需要改 `probe/backends.py` 里的 `VJEPA2Backend.forward`。**故意没用 HF 的 `VJEPA2Model`：它的预测器读取的是编码器在完整片段上的输出，会泄漏目标区域的信息。

## 三、数据清单

```bash
# Hefest 上已确认的 Magic Videos 根目录：/data/datasets/shuang/magic
# 本机 MSYS2：先传到 datakom 的共享 home
scp "/e/PhD/Deepfake Video reconstruction 2026_11_05/predictability_probe/slurm/"*.sbatch \
  slurm-datakom:/home/shuang/

# 在 slurm-datakom 上：复制到项目目录后提交
cp ~/build_magic_manifest.sbatch ~/probe.sbatch ~/standardize.sbatch \
  "/home/shuang/projects/Deepfake Reconstruction/predictability_probe/slurm/"
mkdir -p ~/logs
cd "/home/shuang/projects/Deepfake Reconstruction/predictability_probe"
sbatch slurm/build_magic_manifest.sbatch
```
- 清单脚本的每个 `--spec` 对应一个实际子目录。真实视频目录是 `real/real_videos_pexels_720p` 与 `real/real_videos_mixkit_720p`；假视频目录是 `fake/step-video`、`fake/jimeng-S3.0`、`fake/jimeng-S2.0`、`fake/wanx2.1`、`fake/hailuo-T2V-01`、`fake/pexels_wan2.1-T2V-1.3B`。`root=` 必须使用这些实际路径。
- Hefest 文件名抽样确认：Mixkit 的真实视频与 `step-video`、`jimeng-S3.0`、`jimeng-S2.0`、`wanx2.1`、`hailuo-T2V-01` 使用相同编号（如 `134.mp4`）；Pexels 的真实视频与 `pexels_wan2.1-T2V-1.3B` 使用相同哈希文件名。因此这八个目录都使用 `pair=stem`。脚本会验证每个生成器的配对率，生成后仍须人工打开报告中的配对样例核对。
- 真实视频默认按 id 哈希对半分成 calib（定阈值）和 test（测试）；未配对的假视频进 test，已配对的假视频跟随对应真实视频的划分。
- **有配对关系的数据集一定要加 `pair=...`**：Magic Videos 已按抽样文件名选用 `pair=stem`；CoCoVideo 的规则仍待拿到数据后确定。其他可选规则是 `pair=regex:(clip\d+)` 或 `pair=csv:映射表.csv`。加了以后会**按对来划分**：真实视频和由它生成的假视频总是在同一边；配对的真实视频进了 calib 的那些假视频不参与评估。
  - **配对检查**：如果一个 pair_id 对应多条真实视频，或者某个生成器的配对率低于 `--min-pair-rate`（默认 90%），脚本会**直接报错退出**，不生成清单。找不到对应真实视频的假视频没有 pair_id，一律放进 test 参与评估，不会被丢掉。
  - 脚本会打印每个生成器的配对率，以及 3 组配对示例的路径。**跑探针之前，先人工打开这几组视频确认确实是一对。**
- 脚本最后会打印**每个生成器的分辨率、帧率、编码器分布**，先看一眼有没有明显的格式差异。

统一格式的版本：`sbatch --export=ALL,MANIFEST=manifests/magic_raw.csv,TAG=magic slurm/standardize.sbatch`，会生成 `manifests/magic_std.csv`，视频输出到 `/data/checkpoints/shuang/std_videos`，因为 `/data/datasets` 是只读挂载。

## 四、运行

```bash
# 先用 50 条视频试跑（--limit 会在真实视频和各生成器之间均匀抽样，每组约 7 条）
sbatch --export=ALL,BACKEND=vjepa2,MANIFEST=manifests/magic_raw.csv,TAG=magic_raw_try,EXTRA="--limit 50" slurm/probe.sbatch
# selftest 没全部通过时会以非零状态退出，可以写成：selftest ... && sbatch ...
# 全量：2 个模型 × 2 套数据（原始 / 统一格式）
for B in vjepa2 videomae; do for V in raw std; do
  sbatch --export=ALL,BACKEND=$B,MANIFEST=manifests/magic_$V.csv,TAG=magic_$V slurm/probe.sbatch
done; done
```
默认参数：16 帧，按 8fps 采样（约 1.9 秒）；每条视频最多 4 个片段；跳过开头 0.5 秒（CoCoVideo 的假视频首帧就是真实视频的首帧）；三块裁剪；随机遮挡用 3 个种子。
默认遮挡方式：V-JEPA 用 `future,block:0.25`；VideoMAE 用 `tube:0.9,future`。
**断点续跑**：每条视频的全部结果写完之后，会追加一行"完成标记"。重新启动时，没有完成标记的视频会被清掉、重新计算。结果文件的第一行记录了本次运行的配置；续跑时如果配置不一致（包括 `--no-amp`），程序会直接拒绝。每条视频的完成标记还记录了文件路径和大小；如果同一个 video_id 现在指向另一个文件（比如原始版和统一格式版共用一个输出文件），程序也会拒绝。想换配置或换数据，请用新的 `--out` 文件。往清单里**新增**视频后续跑是允许的。

## 五、分析

```bash
python3 scripts/analyze.py --results /data/checkpoints/shuang/predictability_probe/results/vjepa2_magic_raw.jsonl \
    --manifest manifests/magic_raw.csv --out analysis/vjepa2_magic_raw
```
输出 `summary.md`、`auc.csv`、`confounds.csv`，以及分布图和时间误差曲线。

**怎么读结果**
- `high`：假视频更难预测；`low`：假视频更容易预测；`two`：两端都算异常。前缀 `r_` 表示扣除运动量和纹理复杂度之后的版本，**重点看这个**。
- 先看 `confounds.csv`：如果 bpp（每像素码率）、fps、分辨率这些格式特征单独就能分开真假（>0.8），说明分数可能是借这些格式特征分出来的，要以统一格式版本的结果为准。
- 原始版和统一格式版的结论方向一致、每个生成器的 AUC 都明显大于 0.5，才算"信号存在"。
- **有配对时，以 paired win-rate（假视频的分数高于它自己那条真实视频的概率）为准**。不配对的 AUC 明显高于配对胜率时，说明有一部分差异来自生成器所用 prompt 的内容，而不是真假本身。
- 判断依据是：在多个生成器上方向一致，置信区间不跨 0.5。不要只看 ALL_FAKES 一项：不同生成器的方向可能相反，平均下来会互相抵消。
