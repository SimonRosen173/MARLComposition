import numpy as np
from GridWorld import *
from library import *


def follow_policies(env: MA4RoomsWrapper, Q, joint_start_states, log_file_path, max_steps=100, is_printing=True):
    with open(log_file_path, "w") as f:
        for i, joint_start_state in enumerate(joint_start_states):
            if is_printing:
                print(f"Following policy [Path {i+1}/{len(joint_start_states)}]...")
            follow_extended_q_policy(env, Q, joint_start_state=joint_start_state, is_rendering=False, max_steps=max_steps)
            traj_str = env.get_trajectory(is_delimited=True)
            # print(f"{len(traj_str)}\n\n")
            f.write(traj_str + "\n")

