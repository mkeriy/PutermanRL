from gym.envs.registration import load_env_plugins as _load_env_plugins
from gym.envs.registration import make, register, registry, spec

# Hook to load plugins from entry points
_load_env_plugins()


# Classic
# ----------------------------------------

max_episode_steps = int((1 << 31) - 1)
register(
    id="LongCartPole-v0",
    entry_point="gym.envs.classic_control.cartpole:CartPoleEnv",
    max_episode_steps=max_episode_steps,
    reward_threshold=195.0,
)

register(
    id="LongCartPole-v1",
    entry_point="gym.envs.classic_control.cartpole:CartPoleEnv",
    max_episode_steps=max_episode_steps,
    reward_threshold=475.0,
)

register(
    id="LongMountainCar-v0",
    entry_point="gym.envs.classic_control.mountain_car:MountainCarEnv",
    max_episode_steps=max_episode_steps,
    reward_threshold=-110.0,
)

register(
    id="LongMountainCarContinuous-v0",
    entry_point="gym.envs.classic_control.continuous_mountain_car:Continuous_MountainCarEnv",
    max_episode_steps=max_episode_steps,
    reward_threshold=90.0,
)

register(
    id="LongPendulum-v1",
    entry_point="gym.envs.classic_control.pendulum:PendulumEnv",
    max_episode_steps=max_episode_steps,
)

register(
    id="LongAcrobot-v1",
    entry_point="gym.envs.classic_control.acrobot:AcrobotEnv",
    reward_threshold=-100.0,
    max_episode_steps=max_episode_steps,
)

# Box2d
# ----------------------------------------

register(
    id="LongLunarLander-v2",
    entry_point="gym.envs.box2d.lunar_lander:LunarLander",
    max_episode_steps=max_episode_steps,
    reward_threshold=200,
)

register(
    id="LongLunarLanderContinuous-v2",
    entry_point="gym.envs.box2d.lunar_lander:LunarLander",
    kwargs={"continuous": True},
    max_episode_steps=max_episode_steps,
    reward_threshold=200,
)

register(
    id="LongBipedalWalker-v3",
    entry_point="gym.envs.box2d.bipedal_walker:BipedalWalker",
    max_episode_steps=max_episode_steps,
    reward_threshold=300,
)

register(
    id="LongBipedalWalkerHardcore-v3",
    entry_point="gym.envs.box2d.bipedal_walker:BipedalWalker",
    kwargs={"hardcore": True},
    max_episode_steps=max_episode_steps,
    reward_threshold=300,
)

register(
    id="LongCarRacing-v2",
    entry_point="gym.envs.box2d.car_racing:CarRacing",
    max_episode_steps=max_episode_steps,
    reward_threshold=900,
)

# Toy Text
# ----------------------------------------

register(
    id="LongBlackjack-v1",
    entry_point="gym.envs.toy_text.blackjack:BlackjackEnv",
    kwargs={"sab": True, "natural": False},
)

register(
    id="LongFrozenLake-v1",
    entry_point="gym.envs.toy_text.frozen_lake:FrozenLakeEnv",
    kwargs={"map_name": "4x4"},
    max_episode_steps=max_episode_steps,
    reward_threshold=0.70,  # optimum = 0.74
)

register(
    id="LongFrozenLake8x8-v1",
    entry_point="gym.envs.toy_text.frozen_lake:FrozenLakeEnv",
    kwargs={"map_name": "8x8"},
    max_episode_steps=max_episode_steps,
    reward_threshold=0.85,  # optimum = 0.91
)

register(
    id="LongCliffWalking-v0",
    entry_point="gym.envs.toy_text.cliffwalking:CliffWalkingEnv",
)

register(
    id="LongTaxi-v3",
    entry_point="gym.envs.toy_text.taxi:TaxiEnv",
    reward_threshold=8,  # optimum = 8.46
    max_episode_steps=max_episode_steps,
)

# Mujoco
# ----------------------------------------

# 2D

register(
    id="LongReacher-v2",
    entry_point="gym.envs.mujoco:ReacherEnv",
    max_episode_steps=max_episode_steps,
    reward_threshold=-3.75,
)

register(
    id="LongReacher-v4",
    entry_point="gym.envs.mujoco.reacher_v4:ReacherEnv",
    max_episode_steps=max_episode_steps,
    reward_threshold=-3.75,
)

register(
    id="LongPusher-v2",
    entry_point="gym.envs.mujoco:PusherEnv",
    max_episode_steps=max_episode_steps,
    reward_threshold=0.0,
)

register(
    id="LongPusher-v4",
    entry_point="gym.envs.mujoco.pusher_v4:PusherEnv",
    max_episode_steps=max_episode_steps,
    reward_threshold=0.0,
)

register(
    id="LongInvertedPendulum-v2",
    entry_point="gym.envs.mujoco:InvertedPendulumEnv",
    max_episode_steps=max_episode_steps,
    reward_threshold=950.0,
)

register(
    id="LongInvertedPendulum-v4",
    entry_point="gym.envs.mujoco.inverted_pendulum_v4:InvertedPendulumEnv",
    max_episode_steps=max_episode_steps,
    reward_threshold=950.0,
)

register(
    id="LongInvertedDoublePendulum-v2",
    entry_point="gym.envs.mujoco:InvertedDoublePendulumEnv",
    max_episode_steps=max_episode_steps,
    reward_threshold=9100.0,
)

register(
    id="LongInvertedDoublePendulum-v4",
    entry_point="gym.envs.mujoco.inverted_double_pendulum_v4:InvertedDoublePendulumEnv",
    max_episode_steps=max_episode_steps,
    reward_threshold=9100.0,
)

register(
    id="LongHalfCheetah-v2",
    entry_point="gym.envs.mujoco:HalfCheetahEnv",
    max_episode_steps=max_episode_steps,
    reward_threshold=4800.0,
)

register(
    id="LongHalfCheetah-v3",
    entry_point="gym.envs.mujoco.half_cheetah_v3:HalfCheetahEnv",
    max_episode_steps=max_episode_steps,
    reward_threshold=4800.0,
)

register(
    id="LongHalfCheetah-v4",
    entry_point="gym.envs.mujoco.half_cheetah_v4:HalfCheetahEnv",
    max_episode_steps=max_episode_steps,
    reward_threshold=4800.0,
)

register(
    id="LongHopper-v2",
    entry_point="gym.envs.mujoco:HopperEnv",
    max_episode_steps=max_episode_steps,
    reward_threshold=3800.0,
)

register(
    id="LongHopper-v3",
    entry_point="gym.envs.mujoco.hopper_v3:HopperEnv",
    max_episode_steps=max_episode_steps,
    reward_threshold=3800.0,
)

register(
    id="LongHopper-v4",
    entry_point="gym.envs.mujoco.hopper_v4:HopperEnv",
    max_episode_steps=max_episode_steps,
    reward_threshold=3800.0,
)

register(
    id="LongSwimmer-v2",
    entry_point="gym.envs.mujoco:SwimmerEnv",
    max_episode_steps=max_episode_steps,
    reward_threshold=360.0,
)

register(
    id="LongSwimmer-v3",
    entry_point="gym.envs.mujoco.swimmer_v3:SwimmerEnv",
    max_episode_steps=max_episode_steps,
    reward_threshold=360.0,
)

register(
    id="LongSwimmer-v4",
    entry_point="gym.envs.mujoco.swimmer_v4:SwimmerEnv",
    max_episode_steps=max_episode_steps,
    reward_threshold=360.0,
)

register(
    id="LongWalker2d-v2",
    max_episode_steps=max_episode_steps,
    entry_point="gym.envs.mujoco:Walker2dEnv",
)

register(
    id="LongWalker2d-v3",
    max_episode_steps=max_episode_steps,
    entry_point="gym.envs.mujoco.walker2d_v3:Walker2dEnv",
)

register(
    id="LongWalker2d-v4",
    max_episode_steps=max_episode_steps,
    entry_point="gym.envs.mujoco.walker2d_v4:Walker2dEnv",
)

register(
    id="LongAnt-v2",
    entry_point="gym.envs.mujoco:AntEnv",
    max_episode_steps=max_episode_steps,
    reward_threshold=6000.0,
)

register(
    id="LongAnt-v3",
    entry_point="gym.envs.mujoco.ant_v3:AntEnv",
    max_episode_steps=max_episode_steps,
    reward_threshold=6000.0,
)

register(
    id="LongAnt-v4",
    entry_point="gym.envs.mujoco.ant_v4:AntEnv",
    max_episode_steps=max_episode_steps,
    reward_threshold=6000.0,
)

register(
    id="LongHumanoid-v2",
    entry_point="gym.envs.mujoco:HumanoidEnv",
    max_episode_steps=max_episode_steps,
)

register(
    id="LongHumanoid-v3",
    entry_point="gym.envs.mujoco.humanoid_v3:HumanoidEnv",
    max_episode_steps=max_episode_steps,
)

register(
    id="LongHumanoid-v4",
    entry_point="gym.envs.mujoco.humanoid_v4:HumanoidEnv",
    max_episode_steps=max_episode_steps,
)

register(
    id="LongHumanoidStandup-v2",
    entry_point="gym.envs.mujoco:HumanoidStandupEnv",
    max_episode_steps=max_episode_steps,
)

register(
    id="LongHumanoidStandup-v4",
    entry_point="gym.envs.mujoco.humanoidstandup_v4:HumanoidStandupEnv",
    max_episode_steps=max_episode_steps,
)
