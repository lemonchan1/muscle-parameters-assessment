import os
import numpy as np
import nibabel as nib
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import remove_small_objects
import pandas as pd
from tqdm import tqdm
from config import data_path, label_path, result_path
from pandas import ExcelWriter
RATIO_UPPER_LIMIT = 0.8

def filter_ratio(ratio, method='threshold', slice_ratios=None):
    """
    异常值过滤模块
    方法选项: 'threshold'（阈值法）, 'iqr'（箱线图法）
    """
    if method == 'threshold':
        return np.nan if ratio > RATIO_UPPER_LIMIT else ratio
    elif method == 'iqr' and slice_ratios is not None:
        q1 = np.nanpercentile(slice_ratios, 25)
        q3 = np.nanpercentile(slice_ratios, 75)
        iqr = q3 - q1
        upper_bound = q3 + IQR_FACTOR * iqr
        return np.nan if ratio > upper_bound else ratio
    return ratio

def load_slice_ranges():
    """加载切片范围配置"""
    range_file = os.path.join(os.path.dirname(data_path), 'slice_ranges.xlsx')
    if not os.path.exists(range_file):
        raise FileNotFoundError(f"Slice range file {range_file} not found")
    
    df_ranges = pd.read_excel(range_file, dtype={'filename': str})
    return {
        row['filename']: (int(row['start']), int(row['end']))
        for _, row in df_ranges.iterrows()
    }

def validate_slice_range(slice_range, total_slices):
    """验证并修正切片范围"""
    start, end = slice_range
    start = max(0, start)
    end = min(total_slices-1, end)
    return (min(start, end), max(start, end))

def process_images(data_path, label_path, result_path):
    # 加载切片范围配置
    try:
        slice_ranges = load_slice_ranges()
    except Exception as e:
        print(f"Error loading slice ranges: {str(e)}")
        return

    os.makedirs(result_path, exist_ok=True)
    data_files = [f for f in os.listdir(data_path) if f.endswith('.nii.gz')]
    current_file_ratios = []
    
    for data_file in tqdm(data_files, desc="Processing files"):
        # 获取当前文件的切片范围
        file_key = os.path.splitext(data_file)[0].replace('.nii', '')
        if file_key not in slice_ranges:
            print(f"Warning: No slice range found for {data_file}, skipping")
            continue
            
        # 初始化数据结构
        slice_results = []
        volume_data = {}

        # 加载图像数据
        data_img = nib.load(os.path.join(data_path, data_file))
        label_img = nib.load(os.path.join(label_path, data_file))
        data = data_img.get_fdata()
        label = label_img.get_fdata()
        
        total_slices = data.shape[2]
        unique_labels = np.unique(label)[1:]
        
        # 验证切片范围
        raw_start, raw_end = slice_ranges[file_key]
        start_slice, end_slice = validate_slice_range(
            (raw_start, raw_end), total_slices
        )
        
        # 初始化volume数据收集器
        volume_data = {int(lbl): [] for lbl in unique_labels}

        # 处理每个slice
        for slice_idx in range(total_slices):
            in_range = start_slice <= slice_idx <= end_slice
            data_slice = gaussian(data[:, :, slice_idx], sigma=1)
            label_slice = label[:, :, slice_idx]

            for label_value in unique_labels:
                mask = label_slice == label_value
                record = {
                    'Slice': slice_idx,
                    'Label': int(label_value),
                    'Threshold': np.nan,
                    'Ratio': np.nan
                }

                if mask.any():
                    masked_data = data_slice[mask]
                    
                    # 仅在指定范围内收集volume数据
                    if in_range:
                        volume_data[int(label_value)].extend(masked_data.tolist())
                    # 将小于某个阈值的像素块视为无效块
                    try:
                        thresh = threshold_otsu(masked_data)
                        binary = masked_data < thresh
                        binary_cleaned = remove_small_objects(binary, min_size=0)
                        ratio = binary_cleaned.sum() / len(masked_data)
                        
                        # 记录当前ratio用于动态计算
                        current_file_ratios.append(ratio)
        
                        # 应用过滤（这里使用阈值法）
                        ratio = filter_ratio(ratio, method='threshold')
                        
                        # 或者使用动态箱线图法（需取消注释下面一行）
                        # ratio = filter_ratio(ratio, method='iqr', slice_ratios=current_file_ratios)
                        
                        record.update({
                            'Threshold': thresh,
                            'Ratio': ratio
                        })
                    except:
                        pass

                slice_results.append(record)

        # 生成结果表格
        slice_df = pd.DataFrame(slice_results)
        pivot = slice_df.pivot_table(
            index='Slice',
            columns='Label',
            aggfunc='first'
        )
        pivot.columns = [f'label{col[1]}_{col[0].lower()}' for col in pivot.columns]
        pivot = pivot.sort_index(axis=1).reset_index()

        # 生成汇总数据
        summary_data = []
        for lbl, values in volume_data.items():
            try:
                volume_array = np.array(values)
                total_thresh = threshold_otsu(volume_array)
                total_ratio = (volume_array < total_thresh).mean()
            except:
                total_thresh = np.nan
                total_ratio = np.nan
            
            summary_data.append({
                'Label': lbl,
                'Total_Threshold': total_thresh,
                'Total_Ratio': total_ratio,
                'Start_Slice': start_slice,
                'End_Slice': end_slice
            })

        summary_df = pd.DataFrame(summary_data)

        # 保存结果
        output_file = os.path.join(result_path, f"{file_key}_otsu.xlsx")
        with ExcelWriter(output_file) as writer:
            pivot.to_excel(writer, sheet_name='Slice_Results', index=False)
            summary_df.to_excel(writer, sheet_name='Volume_Summary', index=False)

if __name__ == "__main__":
    process_images(data_path, label_path, result_path)
