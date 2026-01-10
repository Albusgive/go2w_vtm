from isaaclab_rl.rsl_rl import RslRlDistillationStudentTeacherRecurrentCfg
from isaaclab_rl.rsl_rl import RslRlDistillationAlgorithmCfg
from isaaclab_rl.rsl_rl import RslRlDistillationRunnerCfg


@configclass
class UnitreeGo2WRoughDistillCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "unitree_go2w_rough"
    empirical_normalization = False
    obs_groups = {
            "policy": ["policy"],
            "teacher": ["teacher"],
        }
    policy = RslRlDistillationStudentTeacherRecurrentCfg(
        init_noise_std=1.0,
        student_hidden_dims=[1024, 512, 256, 128],
        teacher_hidden_dims=[1024, 512, 256, 128],
        activation="elu",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        rnn_type="gru"  # or "gru"
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=5,
        learning_rate=1.0e-4,
        gradient_length=20,
        max_grad_norm=1.0,
    )
