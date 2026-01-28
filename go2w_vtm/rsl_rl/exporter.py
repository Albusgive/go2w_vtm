import torch
import torch.nn as nn
import copy
import os
import numpy as np
import json

class StudentTeacherExporter(nn.Module):
    """
    专门为 StudentTeacherCNN 设计的导出包装器。
    它将 Dict 输入转换为一个打平的单一向量（Flat Vector）。
    """
    def __init__(self, policy, normalizer=None):
        super().__init__()
        # 深度复制模型组件，确保导出过程不影响原始模型
        self.student_mlp = copy.deepcopy(policy.student)
        self.student_cnns = copy.deepcopy(policy.student_cnns)
        self.normalizer = copy.deepcopy(normalizer) if normalizer else nn.Identity()
        
        # 1. 计算 1D 观测（如 policy_normal）的总长度
        self.num_obs_1d = 0
        self.obs_1d_groups = policy.student_obs_groups_1d
        for group in self.obs_1d_groups:
            shape = policy.student_obs_metadata[group]
            self.num_obs_1d += int(np.prod(shape))
        
        # 2. 计算 2D 观测（如 policy_image）的切片位置和形状还原信息
        self.cnn_metadata = []
        current_idx = self.num_obs_1d
        self.obs_2d_groups = policy.student_obs_groups_2d
        
        if self.student_cnns is not None:
            for group in self.obs_2d_groups:
                shape = policy.student_obs_metadata[group] # 原始形状 [C, H, W]
                flat_len = int(np.prod(shape))
                
                self.cnn_metadata.append({
                    'name': group,
                    'shape': tuple(shape),
                    'start': current_idx,
                    'end': current_idx + flat_len
                })
                current_idx += flat_len
        
        # 导出模型预期的打平输入总长度 (例如你看到的 626)
        self.total_input_len = current_idx

    def forward(self, obs_flat: torch.Tensor):
        # A. 提取并标准化 1D 部分
        obs_1d = obs_flat[:, :self.num_obs_1d]
        obs_1d = self.normalizer(obs_1d)
        
        combined_features = [obs_1d]
        
        # B. 提取 2D 部分，还原形状并通过对应的 CNN
        if self.student_cnns is not None:
            for meta in self.cnn_metadata:
                z_raw = obs_flat[:, meta['start']:meta['end']]
                # 将 (Batch, N) 还原为 (Batch, C, H, W)
                z_2d = z_raw.view(-1, *meta['shape'])
                # 通过对应的 CNN 提取特征（例如提取出 128 维）
                cnn_out = self.student_cnns[meta['name']](z_2d)
                combined_features.append(cnn_out)
            
        # C. 拼接所有特征 (1D + CNN特征) 进入 MLP
        # 这里的维度应该匹配 MLP 的 in_features (例如 178)
        final_input = torch.cat(combined_features, dim=-1)
        return self.student_mlp(final_input)

def export_student_teacher(policy, path, filename="student"):
    """
    导出函数：生成模型文件及拼接顺序说明书
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    
    if not hasattr(policy, 'student_obs_metadata'):
        raise AttributeError("错误：请先在 StudentTeacherCNN.__init__ 中添加 self.student_obs_metadata 存储形状信息。")

    # 1. 自动检测设备并实例化包装器
    device = next(policy.student.parameters()).device
    exporter = StudentTeacherExporter(policy, normalizer=getattr(policy, 'student_obs_normalizer', None))
    exporter.to(device)
    exporter.eval()

    # 2. 生成元数据（说明模型每一段索引代表什么）
    metadata = {
        "model_name": filename,
        "total_input_length": exporter.total_input_len,
        "input_segments": []
    }
    
    # 记录 1D 索引
    curr = 0
    for group in exporter.obs_1d_groups:
        size = int(np.prod(policy.student_obs_metadata[group]))
        metadata["input_segments"].append({
            "group": group, "type": "1D", "range": [curr, curr + size], "shape": policy.student_obs_metadata[group]
        })
        curr += size
        
    # 记录 2D 索引
    for group in exporter.obs_2d_groups:
        size = int(np.prod(policy.student_obs_metadata[group]))
        metadata["input_segments"].append({
            "group": group, "type": "2D", "range": [curr, curr + size], "shape": policy.student_obs_metadata[group]
        })
        curr += size

    # 保存说明书
    with open(os.path.join(path, f"{filename}_info.json"), "w") as f:
        json.dump(metadata, f, indent=4)
        
    with open(os.path.join(path, f"{filename}_readme.txt"), "w", encoding="utf-8") as f:
        f.write(f"模型输入总维数: {exporter.total_input_len}\n")
        f.write("拼接顺序与索引范围:\n")
        for seg in metadata["input_segments"]:
            f.write(f"索引 {seg['range']} -> 组名: {seg['group']} ({seg['type']})\n")

    # 3. 准备示例输入进行导出
    example_input = torch.zeros(1, exporter.total_input_len, device=device)

    # 导出 JIT (.pt)
    try:
        with torch.no_grad():
            traced_script = torch.jit.trace(exporter, example_input)
        traced_script.save(os.path.join(path, f"{filename}.pt"))
        print(f"[Success] JIT 导出成功: {os.path.join(path, f'{filename}.pt')}")
    except Exception as e:
        print(f"[Error] JIT 导出失败: {e}")

    # 导出 ONNX (.onnx)
    try:
        torch.onnx.export(
            exporter,
            example_input,
            os.path.join(path, f"{filename}.onnx"),
            export_params=True,
            opset_version=17,
            input_names=['obs'],
            output_names=['actions'],
            dynamic_axes={'obs': {0: 'batch_size'}, 'actions': {0: 'batch_size'}}
        )
        print(f"[Success] ONNX 导出成功: {os.path.join(path, f'{filename}.onnx')}")
    except Exception as e:
        print(f"[Error] ONNX 导出失败: {e}")