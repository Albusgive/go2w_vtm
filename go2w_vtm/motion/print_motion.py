import numpy as np
motion = np.load("warm_up.npz")
print("Keys (labels):", motion.files)
print(motion["fps"])
print(motion["joint_vel"])