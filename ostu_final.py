from typing import Any

import os
import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage
from skimage import exposure, morphology, filters, feature
import matplotlib.pyplot as plt
import openpyxl
from config import data_path, label_path, result_path, dcm_path, script_dir


# 读取slice范围定义文件
range_df = pd.read_excel(os.path.join(script_dir, f'slice_ranges.xlsx'))

def clahe_enhance(image, clip_limit=0.03, nbins=256):
    """对比度受限的自适应直方图均衡（CLAHE）"""

    min_val = np.min(image)
    max_val = np.max(image)
    image = (image - min_val) / (max_val - min_val)
    image = exposure.equalize_adapthist(image, clip_limit=clip_limit, nbins=nbins)
    return image

def process_slice(img_slice, mask_slice, n_thresholds=2):
    """处理单个切片"""

    roi = img_slice.copy()

    # 步骤1: CLAHE增强
    roi_adjust = clahe_enhance(roi)

    # 步骤3: 提取前景区域
    roi_adjust[mask_slice == 0] = 0  # 背景设为0

    # 步骤4: Otsu多阈值分割

    # 获取前景区域的像素值
    foreground_pixels = roi_adjust[roi_adjust > 0]

    # 检查唯一值的数量是否足够进行多阈值分割
    unique_vals = np.unique(foreground_pixels)

    if len(unique_vals) <= 1:
        # 如果只有1个值，直接创建二值图像（所有前景为同一类）
        regions = np.zeros_like(roi_adjust, dtype=int)
        regions[roi_adjust > 0] = 1
        thresholds = [0, 0]  # 重复阈值以保持输出格式

    elif len(unique_vals) == 2:
        # 如果有2个值，使用单阈值Otsu
        threshold = filters.threshold_otsu(foreground_pixels)
        regions = np.digitize(roi_adjust, bins=[threshold])
        thresholds = [threshold, threshold]  # 重复阈值以保持输出格式一致

    else:
    # 正常的多阈值分割
        try:
            # 尝试增加nbins参数
            thresholds = filters.threshold_multiotsu(foreground_pixels, classes=n_thresholds + 1, nbins=512)
            regions = np.digitize(roi_adjust, bins=thresholds)
        except ValueError as e:
            # 如果多阈值失败，回退到单阈值
            threshold = filters.threshold_otsu(foreground_pixels)
            regions = np.digitize(roi_adjust, bins=[threshold])
            thresholds = [threshold, threshold]


    # 步骤5: 反转二值图像 (0变1，非0变0)
    binary = np.zeros_like(regions, dtype=np.uint8)
    regions[mask_slice == 0] = -1
    binary[regions == 0] = 1  # 背景区域设为1
    binary[regions > 0] = 0  # 分割出的前景设为0

    # 步骤6: 形态学操作序列
    # 小像素噪点清除
    binary = binary > 0
    binary = morphology.remove_small_objects(binary, min_size=10)

    # 6.1 膨胀 (2x2正方形结构元素)
    struct = morphology.footprint_rectangle((2,2))
    dilated = morphology.binary_dilation(binary, struct)

    # 6.2 清除单独像素
    dilated = dilated > 0
    cleaned = morphology.remove_small_objects(dilated, min_size=2)

    # 6.6 填充孔洞
    filled = ndimage.binary_fill_holes(cleaned)

    # # 6.8 顶帽运算
    tophat = morphology.white_tophat(filled, morphology.disk(1))
    final_binary = filled ^ tophat
    final_binary[final_binary < 0] = 0  # 确保无负值

    return thresholds, final_binary


def calculate_label_ratios(final_binary, mask_slice, label_ids):
    """计算各label区域中前景像素比例"""
    ratios = {}
    for label_id in label_ids:
        label_mask = (mask_slice == label_id)
        if np.any(label_mask):
            label_area = np.sum(label_mask)
            label_foreground = np.sum(final_binary[label_mask])
            ratio = label_foreground / label_area
        else:
            ratio = "N/A"
        ratios[label_id] = ratio
    return ratios


# 主处理流程
def main():

    datalist = os.listdir(os.path.join(script_dir, 'data'))
    datalist = [i.replace('.nii.gz', '') for i in datalist]
    # 遍历每个文件
    for data in datalist:
        # 在DataFrame中查找匹配的行
        matched_rows = range_df[range_df['filename'] == data]
        if not matched_rows.empty:
            start_slice = matched_rows.iloc[0]['start']
            end_slice = matched_rows.iloc[0]['end']
            filename = data
            # 加载NIfTI文件
            img_path = os.path.join(script_dir, 'data', filename + '.nii.gz')
            label_path = os.path.join(script_dir, 'label', filename + '.nii.gz')

            img_nii = nib.load(img_path)
            label_nii = nib.load(label_path)

            img_data = img_nii.get_fdata()
            mask_data = label_nii.get_fdata()

            # 准备结果容器
            slice_results = []
            label_accumulator = {i: {'foreground': 0, 'total': 0} for i in range(1, 9)}

            # 处理每个切片
            slice_idx: Any
            for slice_idx in range(start_slice, end_slice):
                img_slice = img_data[:, :, slice_idx]
                mask_slice = mask_data[:, :, slice_idx]

                # 处理当前切片
                thresholds, final_binary = process_slice(img_slice, mask_slice)

                # 计算各label比例
                label_ratios = calculate_label_ratios(final_binary, mask_slice, range(1, 9))

                # 记录结果
                result_row = {
                    'Slice': slice_idx,
                    'threshold1': thresholds[0],
                    'threshold2': thresholds[1] if len(thresholds) > 1 else None
                }
                for label_id in range(1, 9):
                    result_row[f'label{label_id}'] = label_ratios[label_id]

                    # 累加统计信息
                    label_mask = (mask_slice == label_id)
                    if np.any(label_mask):
                        label_accumulator[label_id]['foreground'] += np.sum(final_binary[label_mask])
                        label_accumulator[label_id]['total'] += np.sum(label_mask)

                slice_results.append(result_row)

            # 创建结果DataFrame
            result_df = pd.DataFrame(slice_results)

            # 创建整体统计表
            summary_data = []
            for label_id in range(1, 9):
                if label_accumulator[label_id]['total'] > 0:
                    ratio = label_accumulator[label_id]['foreground'] / label_accumulator[label_id]['total']
                else:
                    ratio = 0.0
                summary_data.append({'Label': f'label{label_id}', 'Ratio': ratio})

            summary_df = pd.DataFrame(summary_data)

            # 保存结果到Excel
            output_filename = os.path.join(script_dir, 'result', f"{filename}_otsu.xlsx")
            with pd.ExcelWriter(output_filename) as writer:
                result_df.to_excel(writer, sheet_name='Slice_Results', index=False)
                summary_df.to_excel(writer, sheet_name='Volume_Summary', index=False)
        # 不存在匹配结果时，直接对所有层进行计算
        else:
            filename = data
            # 加载NIfTI文件
            img_path = os.path.join(script_dir, 'data', filename + '.nii.gz')
            label_path = os.path.join(script_dir, 'label', filename + '.nii.gz')

            img_nii = nib.load(img_path)
            label_nii = nib.load(label_path)

            img_data = img_nii.get_fdata()
            mask_data = label_nii.get_fdata()

            # 准备结果容器
            slice_results = []
            label_accumulator = {i: {'foreground': 0, 'total': 0} for i in range(1, 9)}

            image = sitk.ReadImage(label_path)
            depth = image.GetDepth()

            # 处理每个切片
            for slice_idx in range(0, depth - 1):
                img_slice = img_data[:, :, slice_idx]
                mask_slice = mask_data[:, :, slice_idx]

                # 处理当前切片
                thresholds, final_binary = process_slice(img_slice, mask_slice)

                # 计算各label比例
                label_ratios = calculate_label_ratios(final_binary, mask_slice, range(1, 9))

                # 记录结果
                result_row = {
                    'Slice': slice_idx,
                    'threshold1': thresholds[0],
                    'threshold2': thresholds[1] if len(thresholds) > 1 else None
                }
                for label_id in range(1, 9):
                    result_row[f'label{label_id}'] = label_ratios[label_id]

                    # 累加统计信息
                    label_mask = (mask_slice == label_id)
                    if np.any(label_mask):
                        label_accumulator[label_id]['foreground'] += np.sum(final_binary[label_mask])
                        label_accumulator[label_id]['total'] += np.sum(label_mask)

                slice_results.append(result_row)

            # 创建结果DataFrame
            result_df = pd.DataFrame(slice_results)

            # 创建整体统计表
            summary_data = []
            for label_id in range(1, 9):
                if label_accumulator[label_id]['total'] > 0:
                    ratio = label_accumulator[label_id]['foreground'] / label_accumulator[label_id]['total']
                else:
                    ratio = 0.0
                summary_data.append({'Label': f'label{label_id}', 'Ratio': ratio})

            summary_df = pd.DataFrame(summary_data)

            # 保存结果到Excel
            output_filename = os.path.join(script_dir, 'result', f"{filename}_otsu.xlsx")
            with pd.ExcelWriter(output_filename) as writer:
                result_df.to_excel(writer, sheet_name='Slice_Results', index=False)
                summary_df.to_excel(writer, sheet_name='Volume_Summary', index=False)
        print(f"{filename} 阈值计算已完成")



if __name__ == "__main__":
    main()
