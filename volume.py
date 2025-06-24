import os
import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path
from config import label_path, result_path

def load_slice_ranges(config_excel):
    try:
        df = pd.read_excel(config_excel, dtype={'filename': str, 'start': int, 'end': int})
        # 预处理文件名：移除可能的.nii.gz/.nii后缀
        df['filename'] = df['filename'].str.replace(r'\.nii(\.gz)?$', '', regex=True)
        return df.set_index('filename').to_dict(orient='index')
    except FileNotFoundError:
        print(f"错误：配置文件 {config_excel} 不存在")
        return {}
    except KeyError as e:
        print(f"配置文件列名错误，必须包含 ['filename', 'start', 'end']，缺失列：{str(e)}")
        return {}
    except Exception as e:
        print(f"读取配置文件失败: {str(e)}")
        return {}

def process_single_file(label_file, result_dir, slice_ranges):
    try:
        img = nib.load(label_file)
        data = img.get_fdata()
        zooms = img.header.get_zooms()
        
        # ========== 主要修改部分 ==========
        # 获取基准文件名（去除所有.nii后缀）
        file_path = Path(label_file)
        base_name = file_path.name.replace('.nii.gz', '').replace('.nii', '')
        
        # 获取切片范围配置
        file_config = slice_ranges.get(base_name, {})
        # ================================
        
        slice_range = (file_config.get('start'), file_config.get('end'))

        area_df_all, total_slices = calculate_slice_metrics(data, zooms[:2], slice_range=None)

        if slice_range[0] is not None and slice_range[1] is not None:
            start = max(0, slice_range[0])
            end = min(data.shape[2]-1, slice_range[1])
            volume_area_df = area_df_all[(area_df_all['slice'] >= start) & (area_df_all['slice'] <= end)]
            valid_volume_slices = end - start + 1
        else:
            volume_area_df = area_df_all
            valid_volume_slices = total_slices

        volume_series = calculate_total_volumes(volume_area_df, zooms[2])

        output_path = Path(result_dir) / f"{base_name}.xlsx"
        with pd.ExcelWriter(output_path) as writer:
            area_df_all.to_excel(writer, sheet_name="切片面积", index=False)
            pd.DataFrame({
                "Label": [f"label{i}" for i in range(9)],
                "总体积(mm³)": volume_series.values
            }).to_excel(writer, sheet_name="总体积", index=False)
            
        print(f"成功处理: {base_name} | 面积切片数: {total_slices} | 体积切片数: {valid_volume_slices}")

    except Exception as e:
        print(f"处理失败: {file_path.name} | 错误: {str(e)}")

def calculate_slice_metrics(data, zooms_2d, slice_range=None):
    """（保持原函数不变）"""
    dx, dy = zooms_2d
    pixel_area = dx * dy
    num_slices = data.shape[2]

    if slice_range:
        start = max(0, slice_range[0])
        end = min(num_slices-1, slice_range[1])
        slices = range(start, end+1)
    else:
        slices = range(num_slices)

    results = []
    for z in slices:
        slice_data = data[:, :, z]
        row = {"slice": z}
        for label in range(9):
            mask = (slice_data == label)
            row[f"label{label}_area_mm2"] = np.sum(mask) * pixel_area
        results.append(row)
    return pd.DataFrame(results), len(slices)

def calculate_total_volumes(area_df, dz):
    """（保持原函数不变）"""
    area_cols = [col for col in area_df if col.endswith("_area_mm2")]
    total_areas = area_df[area_cols].sum()
    return total_areas * dz

def batch_process(label_path, result_path, config_excel=None):
    """（保持原函数不变）"""
    slice_ranges = load_slice_ranges(config_excel) if config_excel else {}
    
    for root, _, files in os.walk(label_path):
        for file in files:
            if file.lower().endswith((".nii.gz", ".nii")):
                process_single_file(
                    label_file=os.path.join(root, file),
                    result_dir=result_path,
                    slice_ranges=slice_ranges
                )

if __name__ == "__main__":
    config_excel = Path(__file__).parent/"slice_ranges.xlsx"
    batch_process(label_path, result_path, config_excel=config_excel)

