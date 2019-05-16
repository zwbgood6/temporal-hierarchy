from gym.envs.registration import register

register(
    id='CustomPush-v0',
    entry_point='gym_push.envs:CustomPushEnv',
)

register(
    id="TwoObjectPush-v0",
    entry_point='gym_push.envs:TwoObjectPushEnv',
)

register(
    id="SingleObjectPush-v0",
    entry_point='gym_push.envs:SingleObjectPushEnv',
)

register(
    id="ObstaclePush-v0",
    entry_point='gym_push.envs:ObstaclePushEnv',
)

register(
    id="MultigoalObstaclePush-v0",
    entry_point='gym_push.envs:MultigoalObstaclePushEnv',
)
