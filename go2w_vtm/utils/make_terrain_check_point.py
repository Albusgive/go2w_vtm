#对于不同地形计算不同的checkpoint
import numpy as np

def makeBoxTrenchTerrainCheckPoint(
    difficulty: float,size: tuple[float, float],trench_x: float,trench_width: tuple[float, float],robot_origin_x: float,
    init_key_pos_distance: float,end_key_pos_distance: float)->list[np.ndarray]:
    
    if init_key_pos_distance is None or end_key_pos_distance is None:
        raise ValueError("init_key_pos_distance and end_key_pos_distance must be provided")
    
    trench_x_min= robot_origin_x + trench_x
    trench_width = trench_width[0] + (trench_width[1] - trench_width[0]) * difficulty
    trench_x_max = trench_x_min + trench_width
    
    terrain_key_pos_list = []
    
    init_key_pos = np.array([trench_x_min - init_key_pos_distance, size[1]/2, 0])
    terrain_key_pos_list.append(init_key_pos)

    # 中间的checkpoint
    # 1.沟壑边缘 并脚
    middle_key_pos = np.array([trench_x_min, size[1]/2, 0])
    terrain_key_pos_list.append(middle_key_pos)
    
    # 2.沟壑边缘 准备发力
    middle_key_pos = np.array([trench_x_min + 0.1, size[1]/2, 0])
    terrain_key_pos_list.append(middle_key_pos)
    
    # 3.沟壑边缘 发力结束
    middle_key_pos = np.array([trench_x_min + 0.2, size[1]/2, 0])
    terrain_key_pos_list.append(middle_key_pos)
    
    # 4.到达另一端
    middle_key_pos = np.array([trench_x_max, size[1]/2, 0])
    terrain_key_pos_list.append(middle_key_pos)
    
    end_key_pos = np.array([trench_x_max + end_key_pos_distance, size[1]/2, 0])
    terrain_key_pos_list.append(end_key_pos)
    
    return terrain_key_pos_list
