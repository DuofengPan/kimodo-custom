# gait-batch — Kimodo 步态库批量生成

基于 [Kimodo](https://github.com/nv-tlabs/kimodo) 的可复用步态 BVH 库构建工具。通过 YAML 配置驱动批量文本生成、root2d 速度约束、NPZ/BVH 导出与质量检查，输出可直接接入 [SOMA Retargeter](https://github.com/NVIDIA/soma-retargeter) 等下游管线。

## 目录结构

```text
gait-batch/
├── configs/                  # 配置文件（可扩展，无需改代码）
│   ├── defaults.yaml         # 模型、导出、输出目录默认值
│   ├── catalog.yaml          # 生成任务清单（jobs）
│   ├── qc.yaml               # 质量检查阈值
│   └── prompts/
│       └── seed_locomotion.yaml   # SEED 风格 prompt 与步态定义
├── gait_batch/               # Python 包
│   ├── catalog.py            # 任务展开（catalog → GenerationTask）
│   ├── constraints.py        # root2d 约束构建
│   ├── generate.py           # 批量生成引擎
│   ├── export.py             # NPZ / BVH / meta 写出
│   ├── qc/                   # 质量检查
│   └── cli.py                # 命令行入口
└── outputs/
    ├── pending/              # 新生成、待 QC
    ├── approved/             # QC 通过（可选移动）
    └── rejected/             # QC 未通过（可选移动）
```

每条成功生成的动作在 `outputs/pending/<gait>/<speed_band>/vNNN/` 下包含：

| 文件 | 说明 |
|------|------|
| `motion.npz` | Kimodo 标准 NPZ（somaskel77） |
| `motion.bvh` | SOMA BVH（默认 BONES-SEED rest pose） |
| `constraints.json` | 使用的 root2d 约束（若有） |
| `meta.json` | prompt、seed、目标速度、QC 状态等元数据 |

## 环境要求

- 已安装 Kimodo（本仓库根目录 `pip install -e .`）
- GPU 推荐；显存不足时可设 `TEXT_ENCODER_DEVICE=cpu`
- 大批量建议单独启动 text encoder 服务：

```bash
# 终端 1
kimodo_textencoder

# 终端 2
export TEXT_ENCODER_MODE=api
```

## 快速开始

在 **kimodo 仓库根目录** 或 **gait-batch 目录** 下执行：

```bash
cd gait-batch

# 1. 查看 catalog 展开后的全部任务（当前默认 94 条）
python -m gait_batch list-jobs

# 2. 试跑：只列出前 3 条，不调用模型
python -m gait_batch generate --dry-run --limit 3

# 3. 实际生成（跳过已存在条目，支持断点续跑）
python -m gait_batch generate

# 4. 仅生成某一类步态
python -m gait_batch generate --gait walk_forward

# 5. 质量检查
python -m gait_batch qc --input outputs/pending

# 6. QC 通过后自动分拣（可选）
python -m gait_batch qc --input outputs/pending \
  --move-on-pass \
  --approved outputs/approved \
  --rejected outputs/rejected
```

## 配置说明

### 增删步态类别

编辑 `configs/catalog.yaml` 的 `jobs` 列表：

```yaml
jobs:
  - gait: walk_forward
    speed_bands: [slow, medium, fast]
    num_variants: 8          # 每档速度 8 条变体
  - gait: walk_left
    speed_bands: [medium]
    num_variants: 4
```

### 新增 prompt / 步态类型

在 `configs/prompts/seed_locomotion.yaml` 增加 `gait_types` 条目，并在 `catalog.yaml` 的 `jobs` 中引用：

```yaml
gait_types:
  my_custom_gait:
    motion:
      type: linear
      direction: [0.0, 1.0]    # 地面 [x, z]，+z 为前进
      use_root2d: true
      density: sparse          # sparse: 起终点；dense: 每帧
    prompts:
      medium: "A person walks forward while looking around cautiously."
```

Prompt 写法建议参考 [BONES-SEED](https://huggingface.co/datasets/bones-studio/seed)：`"A person..."` 开头、中等细节、单行为主。详见 Kimodo [best practices](https://research.nvidia.com/labs/sil/projects/kimodo/docs/key_concepts/limitations.html)。

### 速度分档（root2d）

在 `configs/prompts/seed_locomotion.yaml` 的 `speed_bands` 中定义目标速度（m/s）：

```yaml
speed_bands:
  slow:
    root_speed_mps: 0.45
  medium:
    root_speed_mps: 0.85
```

对启用 `use_root2d: true` 的步态，工具自动计算：

`root_distance_m = root_speed_mps × duration_sec`

并生成 sparse root2d 约束（帧 0 → 最后一帧），与文本描述共同约束 locomotion 速度。

原地转向等不需要 root 位移的步态设 `use_root2d: false`。

### 全局参数

`configs/defaults.yaml` 可调整模型、时长、batch 大小、CFG、BVH rest pose 等：

```yaml
generation:
  model: kimodo-soma-rp-v1.1
  duration_sec: 6.0
  batch_size: 4              # 增大可加速；精确 per-task seed 复现建议设为 1
  post_processing: true

export:
  bvh_standard_tpose: false    # false = BONES-SEED rest pose（与 SEED BVH 一致）
```

### QC 阈值

编辑 `configs/qc.yaml`：最小 root 位移、速度容差、foot skate 上限等。QC 报告写入 `outputs/pending/qc_report.json`。

## 设计原则

- **配置驱动**：增删步态、改 prompt、改速度档、改变体数量均只改 YAML
- **模块化**：catalog / constraints / generate / export / qc 各自独立，便于接入自定义 QC 或导出格式
- **断点续跑**：已存在 `motion.npz` + `motion.bvh` + `meta.json` 的条目默认跳过（`--overwrite` 强制重生成）
- **可追溯**：每条 clip 有完整 `meta.json`，含 prompt、seed、目标/实测速度、QC 结果

## 与 SOMA Retargeter 衔接

1. 从 `outputs/approved/`（或 `pending/` 中 QC 通过项）取 `motion.bvh`
2. 确认 retargeter 期望的 rest pose：默认导出为 BONES-SEED 风格；若需要标准 T-pose，在 `defaults.yaml` 设 `bvh_standard_tpose: true`
3. 帧率 30 Hz，与 Kimodo / SEED 管线一致

## 扩展开发

| 需求 | 建议修改位置 |
|------|----------------|
| 新路径类型（弧线、8 字等） | `gait_batch/constraints.py` |
| 新 QC 指标 | `gait_batch/qc/metrics.py` + `configs/qc.yaml` |
| 从 SEED 抽取 prompt 模板 | 新增脚本到 `gait_batch/` 或独立 `scripts/`，输出到 `configs/prompts/` |
| 接入 TMR 文本对齐分数 | 在 `qc/runner.py` 中扩展 |
| 自定义输出命名/格式 | `gait_batch/export.py` |

## 常见问题

**Q: batch_size 与 seed 的关系？**  
同一 batch 内共用 batch 首条的 `seed_everything`（与 benchmark 脚本一致）。batch 内各样本仍因扩散初始噪声不同而有所差异。若需严格 per-task 可复现，将 `batch_size` 设为 `1`。

**Q: 生成很慢？**  
启动 `kimodo_textencoder` 并设 `TEXT_ENCODER_MODE=api`；适当增大 `batch_size`；调试时用 `--limit N`。

**Q: 如何只重跑 QC？**  
直接 `python -m gait_batch qc --input outputs/pending`，无需重新生成。

## License

Apache-2.0（与 Kimodo 主仓库一致）
