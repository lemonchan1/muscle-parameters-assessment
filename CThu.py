import os
import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm
from config import data_path, label_path, result_path

def load_slice_ranges():
    """加载切片范围配置文件"""
    config_path = os.path.join(os.path.dirname(data_path), 'slice_ranges.xlsx')
    try:
        df_ranges = pd.read_excel(config_path, dtype={'filename': str})
        # 修正变量名并清理文件名
        df_ranges['filename'] = df_ranges['filename'].str.replace(
            r'\.nii(\.gz)?$', '', regex=True
        ).str.strip()
        return df_ranges.set_index('filename').to_dict(orient='index')
    except FileNotFoundError:
        print(f"错误：配置文件 {config_path} 不存在")
        return {}
    except Exception as e:
        print(f"读取配置文件失败: {str(e)}")
        return {}

def calculate_ct_density():
    os.makedirs(result_path, exist_ok=True)
    slice_ranges = load_slice_ranges()
    
    label_files = [f for f in os.listdir(label_path) if f.endswith(('.nii', '.nii.gz'))]
    
    for label_file in tqdm(label_files, desc="Processing files"):
        # 获取基准文件名（去除所有.nii后缀）
        base_name = os.path.splitext(label_file)[0]
        if base_name.endswith('.nii'):
            base_name = os.path.splitext(base_name)[0]
            
        # 获取切片范围配置
        file_config = slice_ranges.get(base_name)
        if not file_config:
            print(f"警告: {label_file} 未找到切片范围配置，已跳过")
            continue
            
        # 加载数据文件
        data_file = os.path.join(data_path, label_file)
        if not os.path.exists(data_file):
            print(f"警告: {label_file} 对应的数据文件不存在")
            continue
            
        label_img = nib.load(os.path.join(label_path, label_file))
        data_img = nib.load(data_file)
        
        if label_img.shape != data_img.shape:
            print(f"错误: {label_file} 与数据文件维度不匹配")
            continue
            
        label_data = label_img.get_fdata()
        ct_data = data_img.get_fdata()
        unique_labels = np.unique(label_data)[1:]
        
        if len(unique_labels) == 0:
            print(f"警告: {label_file} 未找到有效标签")
            continue

        # 验证切片范围有效性
        total_slices = label_data.shape[2]
        start = max(0, file_config['start'])
        end = min(total_slices-1, file_config['end'])
        if start > end:
            start, end = end, start

        # 初始化存储结构
        slice_results = []
        label_totals = {label: {'sum': 0, 'count': 0} for label in unique_labels}

        # 处理每个切片
        for slice_idx in range(total_slices):
            slice_label = label_data[:, :, slice_idx]
            slice_ct = ct_data[:, :, slice_idx]
            
            slice_dict = {'Slice': slice_idx}
            in_range = start <= slice_idx <= end
            
            for label in unique_labels:
                mask = (slice_label == label)
                if np.any(mask):
                    mean_value = np.mean(slice_ct[mask])
                    slice_dict[f'label{int(label)}'] = mean_value
                    
                    # 仅在指定范围内累加数据
                    if in_range:
                        label_totals[label]['sum'] += mean_value * np.sum(mask)
                        label_totals[label]['count'] += np.sum(mask)
                else:
                    slice_dict[f'label{int(label)}'] = np.nan
                    
            slice_results.append(slice_dict)

        # 生成结果
        df_slice = pd.DataFrame(slice_results)
        
        summary_data = []
        for label in unique_labels:
            if label_totals[label]['count'] > 0:
                total_mean = label_totals[label]['sum'] / label_totals[label]['count']
            else:
                total_mean = np.nan
            summary_data.append({
                'Label': f'label{int(label)}',
                'Total_Mean': total_mean,
                'Start_Slice': start,
                'End_Slice': end
            })
            
        df_summary = pd.DataFrame(summary_data)

        # 保存结果
        output_file = os.path.join(result_path, f"{base_name}_CT.xlsx")
        with pd.ExcelWriter(output_file) as writer:
            df_slice.to_excel(writer, sheet_name='Slice_Means', index=False)
            df_summary.to_excel(writer, sheet_name='Total_Means', index=False)

if __name__ == "__main__":
    calculate_ct_density()
