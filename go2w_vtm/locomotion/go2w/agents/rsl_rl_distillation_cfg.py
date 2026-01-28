from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlDistillationStudentTeacherRecurrentCfg
from isaaclab_rl.rsl_rl import RslRlDistillationAlgorithmCfg
from isaaclab_rl.rsl_rl import RslRlDistillationRunnerCfg


from go2w_vtm.rsl_rl.distillation import RslRlDistillationStudentTeacherCNNCfg

@configclass
class UnitreeGo2WRoughMultiDistillCNNCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 500
    experiment_name = "unitree_go2w_rough_multi_motion_distill_cnn"
    empirical_normalization = True # Runner 层面的 norm 通常关掉，由 Policy 内部控制

    obs_groups = {
            "policy": ["policy_normal",
                       "policy_image",],   # Student 观测 因为包含图像要单独处理所以需要设定policy组
            "teacher": ["teacher"], # Teacher 观测 因为lab拉平了所以key名为 teacher
        }

    policy = RslRlDistillationStudentTeacherCNNCfg(
        init_noise_std=1.0,
        # Student 参数
        student_hidden_dims=[512, 256, 128],
        student_obs_normalization=False, # CNN 输入通常不需要 Empirical Norm，如果是混合输入，代码内部只对 MLP 部分做 Norm
        
        # Teacher 参数
        teacher_hidden_dims=[512, 256, 128],
        teacher_obs_normalization=False,  # <--- 修正点：如果 Teacher 训练时开了 Norm，这里必须为 True
        
        activation="elu",
        
        # CNN 参数
        student_cnn_cfg = {
            "policy_image": {
                "output_channels": [32, 64, 128],
                "kernel_size": [3, 3, 3],
                "stride": [2, 1, 1],          
                "padding": "zeros", 
                "norm": "batch",
                "activation": "elu",
                "max_pool": [False, False, False], 
                "global_pool": "avg",         
                "flatten": True,              
            }
        },
        teacher_cnn_cfg=None, # 纯 MLP Teacher
    )
    
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=5,
        learning_rate=1.0e-4,
        gradient_length=20,
        max_grad_norm=1.0,
    )