# DDM² Training Notebooks

## 概述

DDM² (Denoising Diffusion Models for Medical imaging) 训练流程:

```
External N2N (Teacher) → Stage 2 (State Matching) → Stage 3 (Train Diffusion)
```

- **Stage 2**: 为每个样本找最优扩散时间步 t
- **Stage 3**: 用匹配结果训练扩散去噪模型

## 数据结构

```
数据组织层级:
├── batch      # 一组病例，用于划分训练/验证集
│   ├── volume (case)  # 一个病例，即一个完整的3D CT扫描
│   │   ├── slice 0    # 2D切片
│   │   ├── slice 1
│   │   └── ...
```

- `volume` = `case` = 一个病人的一次CT扫描
- `slice` = volume中的一层2D图像
- `volume_idx=3, slice_idx=45` 表示第3个病例的第45层

## 使用方法

1. 修改第一个cell里的配置参数
2. 运行所有cell

## 执行顺序

```
01_stage2_state_matching.ipynb  →  02_stage3_train_diffusion.ipynb  →  03_inference.ipynb  →  04_quantitative_evaluation.ipynb
        (状态匹配)                       (训练扩散模型)                    (推理)                    (量化评估)
```

## 主要配置说明

### 路径配置
| 参数 | 说明 |
|------|------|
| `PROJECT_ROOT` | 项目根目录 |
| `EXCEL_PATH` | 数据索引Excel文件 |
| `DATA_ROOT` | CT数据根目录 |
| `TEACHER_N2N_ROOT` | Teacher N2N预测结果 |

### 数据划分
| 参数 | 说明 |
|------|------|
| `TRAIN_BATCHES` | 训练集batch编号，如 [0,1,2,3,4] |
| `VAL_BATCHES` | 验证集batch编号，如 [5] |
| `SLICE_RANGE` | 切片范围 [start, end)，去掉边缘 |

### 预处理
| 参数 | 说明 |
|------|------|
| `HU_MIN/HU_MAX` | CT值范围，典型[-1000, 2000] |
| `HISTOGRAM_EQUALIZATION` | 直方图均衡化开关 |

### 训练参数 (Stage 3)
| 参数 | 说明 |
|------|------|
| `N_ITER` | 总迭代次数，如100000 |
| `VAL_FREQ` | 验证频率 |
| `SAVE_FREQ` | checkpoint保存频率 |
| `LEARNING_RATE` | 学习率，默认1e-4 |
| `RESUME_STATE` | 断点续训路径 |

### 推理参数 (Inference)
| 参数 | 说明 |
|------|------|
| `CHECKPOINT` | checkpoint路径，None自动查找最新 |
| `INFERENCE_MODE` | "single"单个患者 / "batch"批量 |
| `PATIENT_IDX` | 单个模式时的患者索引 |
| `OUTPUT_DIR` | 输出目录 |
| `INVERSE_HE` | 是否逆向HE转回原始HU空间 |

### 量化评估参数 (Evaluation)
| 参数 | 说明 |
|------|------|
| `DDM2_PRED_ROOT` | DDM² 推理结果目录 |
| `N2N_PRED_ROOT` | Teacher N2N 结果目录 |
| `EVAL_BATCHES` | 评估哪些batch，如 [5] |
| `OUTPUT_EXCEL` | 结果保存路径 |

**评估指标**:
- MAE: Mean Absolute Error (越低越好)
- SSIM: Structural Similarity Index (越高越好)
- LPIPS: Learned Perceptual Image Patch Similarity (越低越好)

## 换电脑时

只需修改第一个cell里的路径配置即可。
