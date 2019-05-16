from gym.envs.registration import register

register(
    id='CustomMove-v0',
    entry_point='gym_move.envs:CustomMoveEnv',
)

