# DDM² Training Notebooks

## 使用流程

1. 运行 `01_stage2_state_matching.ipynb` - 状态匹配
2. 运行 `02_stage3_train_diffusion.ipynb` - 扩散模型训练

## 换电脑时需要修改的配置

在每个 notebook 的第一个 cell 中修改以下路径：

```python
# 项目根目录
PROJECT_ROOT = "/host/d/ddm2"

# 数据路径
EXCEL_PATH = "/host/d/file/fixedCT_static_simulation_train_test_gaussian_local.xlsx"
DATA_ROOT = "/host/d/file/simulation/"

# Teacher N2N 预测结果
TEACHER_N2N_ROOT = "/host/d/file/pre/noise2noise/pred_images/"
TEACHER_N2N_EPOCH = 78

# 直方图均衡化文件
BINS_FILE = "/host/d/file/histogram_equalization/bins.npy"
BINS_MAPPED_FILE = "/host/d/file/histogram_equalization/bins_mapped.npy"
```

## 数据集划分

```python
TRAIN_BATCHES = [0, 1, 2, 3, 4]  # 训练集 (batch 0-4)
VAL_BATCHES = [5]                 # 验证集 (batch 5)
```

## 输出文件

- Stage 2 输出: `{PROJECT_ROOT}/experiments/ct_denoise_stage2/stage2_matched.txt`
- Stage 3 输出: `{PROJECT_ROOT}/experiments/ct_denoise_{timestamp}/`
  - `checkpoint/` - 模型权重
  - `results/` - 验证结果图片
  - `logs/` - 训练日志

## 恢复训练

在 Stage 3 notebook 中设置:
```python
RESUME_STATE = "experiments/ct_denoise_xxx/checkpoint/latest"
```
