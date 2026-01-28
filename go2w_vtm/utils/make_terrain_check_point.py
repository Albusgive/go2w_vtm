#对于不同地形计算不同的checkpoint
from typing import List, Tuple
import numpy as np

def makeBoxTrenchTerrainCheckPoint(
    difficulty: float,size: tuple[float, float],terrain_x: float,robot_origin_x: float,
    trench_width: tuple[float, float])->Tuple[List[np.ndarray], List[str]]:
    
    terrain_x_min= robot_origin_x + terrain_x
    trench_width = trench_width[0] + (trench_width[1] - trench_width[0]) * difficulty
    terrain_x_max = terrain_x_min + trench_width
    
    terrain_key_pos_list = []
    descriptions = []
    
    init_key_pos = np.array([robot_origin_x, size[1]/2, 0])
    terrain_key_pos_list.append(init_key_pos)
    descriptions.append("init")

    # 中间的checkpoint
    # 1.沟壑边缘 并脚
    middle_key_pos = np.array([terrain_x_min, size[1]/2, 0])
    terrain_key_pos_list.append(middle_key_pos)
    descriptions.append("trench_edge")
    
    # 2.沟壑边缘 准备发力
    middle_key_pos = np.array([terrain_x_min + 0.1, size[1]/2, 0])
    terrain_key_pos_list.append(middle_key_pos)
    descriptions.append("trench_edge force")
    
    # 3.沟壑边缘 发力结束
    middle_key_pos = np.array([terrain_x_min + 0.2, size[1]/2, 0])
    terrain_key_pos_list.append(middle_key_pos)
    descriptions.append("trench_edge force end")
    
    # 4.到达另一端
    middle_key_pos = np.array([terrain_x_max, size[1]/2, 0])
    terrain_key_pos_list.append(middle_key_pos)
    descriptions.append("trench other edge")
    
    # 5.离开沟壑
    middle_key_pos = np.array([terrain_x_max + 0.1, size[1]/2, 0])
    terrain_key_pos_list.append(middle_key_pos)
    descriptions.append("trench end")
    
    # 结束位置
    end_key_pos = np.array([terrain_x_max + (size[0]-terrain_x_max)/2, size[1]/2, 0])
    terrain_key_pos_list.append(end_key_pos)
    descriptions.append("end")
    
    return terrain_key_pos_list,descriptions


def makeBoxHighPlatformTerrainCheckPoint(
    difficulty: float,size: tuple[float, float],terrain_x: float,robot_origin_x: float,
    platform_width: float,platform_height: tuple[float, float])->list[np.ndarray]:
    
    
    terrain_x_min= robot_origin_x + terrain_x
    _platform_height = platform_height[0] + (platform_height[1] - platform_height[0]) * difficulty
    
    terrain_key_pos_list = []
    descriptions = []
    
    #初始位置
    init_key_pos = np.array([robot_origin_x, size[1]/2, 0])
    terrain_key_pos_list.append(init_key_pos)
    descriptions.append("init")

    # 中间的checkpoint
    # 1.平台边缘
    middle_key_pos = np.array([terrain_x_min - 0.1, size[1]/2, 0])
    terrain_key_pos_list.append(middle_key_pos)
    descriptions.append("platform_edge")

    # 2.平台边缘 前腿顶住墙
    middle_key_pos = np.array([terrain_x_min - 0.05, size[1]/2, 0.0])
    terrain_key_pos_list.append(middle_key_pos)
    descriptions.append("platform_edge front leg")
    
    # 3.平台边缘 整个身子起来，收一条腿
    middle_key_pos = np.array([terrain_x_min, size[1]/2, 0.0])
    terrain_key_pos_list.append(middle_key_pos)
    descriptions.append("platform_up_edge back one leg up")

    
    # 4.平台边缘 收腿
    middle_key_pos = np.array([terrain_x_min + 0.1, size[1]/2, _platform_height])
    terrain_key_pos_list.append(middle_key_pos)
    descriptions.append("platform_up_edge leg down")
    
    # 5.收一条腿
    middle_key_pos = np.array([terrain_x_min + 0.15, size[1]/2, _platform_height])
    terrain_key_pos_list.append(middle_key_pos)
    descriptions.append("platform_up_edge leg*2 down")
    
    # 6.恢复姿态 恢复正常站姿
    middle_key_pos = np.array([terrain_x_min + 0.2, size[1]/2, _platform_height])
    terrain_key_pos_list.append(middle_key_pos)
    descriptions.append("finished up")

    # 7.结束位置
    end_key_pos = np.array([terrain_x_min + platform_width/2, size[1]/2, _platform_height])
    terrain_key_pos_list.append(end_key_pos)
    descriptions.append("end")
    
    return terrain_key_pos_list,descriptions



def makeBoxRockFissureTerrainCheckPoint(
    difficulty: float,size: tuple[float, float],terrain_x: float,robot_origin_x: float,
    rock_fissure_long: float, rock_fissure_width: tuple[float, float], rock_fissure_height: float)->list[np.ndarray]:
    
    
    terrain_x_min= robot_origin_x + terrain_x
    
    
    terrain_key_pos_list = []
    descriptions = []
    
    init_key_pos = np.array([robot_origin_x, size[1]/2, 0])
    terrain_key_pos_list.append(init_key_pos)
    descriptions.append("init")
    
    _rock_fissure_width = rock_fissure_width[0] + (rock_fissure_width[1] - rock_fissure_width[0]) * difficulty

    # 中间的checkpoint
    # 1.狭缝边缘
    middle_key_pos = np.array([terrain_x_min - 0.1, size[1]/2, 0])
    terrain_key_pos_list.append(middle_key_pos)
    descriptions.append("rock_fissure_edge")

    
    # 2.狭缝边缘 转身
    middle_key_pos = np.array([terrain_x_min, size[1]/2, 0])
    terrain_key_pos_list.append(middle_key_pos)
    descriptions.append("rock_fissure_edge turn")

    
    # 3.狭缝边缘 完全进入
    middle_key_pos = np.array([terrain_x_min + 0.1, size[1]/2, 0])
    terrain_key_pos_list.append(middle_key_pos)
    descriptions.append("rock_fissure_edge enter")

    
    # 4.狭缝出口 转身
    middle_key_pos = np.array([terrain_x_min + rock_fissure_long, size[1]/2, 0])
    terrain_key_pos_list.append(middle_key_pos)
    descriptions.append("rock_fissure_edge turn")
    
    # 4.狭缝出口 完全退出
    middle_key_pos = np.array([terrain_x_min + rock_fissure_long, size[1]/2, 0])
    terrain_key_pos_list.append(middle_key_pos)
    descriptions.append("rock_fissure_edge exit")

    
    # 恢复姿态
    end_key_pos = np.array([terrain_x_min + rock_fissure_long + 0.1, size[1]/2, 0])
    terrain_key_pos_list.append(end_key_pos)
    descriptions.append("end")

    
    return terrain_key_pos_list,descriptions



def makeBoxFloatBoxTerrainCheckPoint(
    difficulty: float,size: tuple[float, float],terrain_x: float,robot_origin_x: float,
    float_box_long: float,float_box_float_height: tuple[float, float],float_box_height: tuple[float, float])->list[np.ndarray]:
    
    _float_box_float_height = float_box_float_height[0] + (float_box_float_height[1] - float_box_float_height[0]) * difficulty
    terrain_x_min= robot_origin_x + terrain_x
    
    
    terrain_key_pos_list = []
    descriptions = []
    
    init_key_pos = np.array([robot_origin_x, size[1]/2, 0])
    terrain_key_pos_list.append(init_key_pos)
    descriptions.append("init")
    

    # 中间的checkpoint
    # 1.float_box边缘
    middle_key_pos = np.array([terrain_x_min - 0.1, size[1]/2, 0])
    terrain_key_pos_list.append(middle_key_pos)
    descriptions.append("float_box_edge")

    
    # 2.float_box边缘 趴下
    middle_key_pos = np.array([terrain_x_min, size[1]/2, 0])
    terrain_key_pos_list.append(middle_key_pos)
    descriptions.append("float_box_edge down")

    
    # 3.float_box末端
    middle_key_pos = np.array([terrain_x_min + float_box_long, size[1]/2, 0])
    terrain_key_pos_list.append(middle_key_pos)
    descriptions.append("float_box_end")
    
    # 4.float_box末端 起身
    middle_key_pos = np.array([terrain_x_min + float_box_long + 0.1, size[1]/2, 0])
    terrain_key_pos_list.append(middle_key_pos)
    descriptions.append("float_box_end up")

    # 恢复姿态
    end_key_pos = np.array([terrain_x_min + float_box_long + 0.2, size[1]/2, 0])
    terrain_key_pos_list.append(end_key_pos)
    descriptions.append("end")

    
    return terrain_key_pos_list,descriptions
