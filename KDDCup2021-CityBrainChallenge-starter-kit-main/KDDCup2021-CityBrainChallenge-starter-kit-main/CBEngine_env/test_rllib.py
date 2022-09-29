from ray import tune
import gym
from env.CBEngine_rllib.CBEngine_rllib import CBEngine_rllib

if __name__ == '__main__':

    env_config = {
        "simulator_cfg_file": 'cfg/simulator.cfg',
        "thread_num": 8,
        "gym_dict": {
            'observation_features':['classic'],
            'reward_feature':'qlength'
        },
        "metric_period" : 3600
    }
    ACTION_SPACE = gym.spaces.Discrete(9)
    OBSERVATION_SPACE = gym.spaces.Dict({
        "observation": gym.spaces.Box(low=-1e10, high=1e10, shape=(48,))
    })
    stop = {
        "training_iteration": 5
    }
    tune_config = {
        "env":CBEngine_rllib,
        "env_config" : env_config,
        "multiagent": {
            "policies": {
                "default_policy": (None, OBSERVATION_SPACE, ACTION_SPACE, {},)
            }
        },

        "lr": 1e-4,
        "log_level": "WARN",
        "lambda": 0.95
    }
    env = CBEngine_rllib(env_config)
    obs = env.reset()
    dones = {}
    dones['__all__'] = False
    step = 0
    while dones['__all__'] == False:
        actions = {}
        for agent in obs.keys():
            actions[agent] = (step//2) % 8 + 1
        obs, rwd, dones, info = env.step(actions)

        # print('step: {}\nobs : {}\nrwd :{}\ndones :{}\n\n\n\n'.format(step,obs,rwd,dones))
        for k,v in obs.items():
            print('step: {}\nobs : {}\nrwd :{}\ndones :{}\n\n\n\n'.format(step,obs[k],rwd[k],dones[k]))
            break
        step+=1
    # tune.run("A3C",config = tune_config,stop = stop)
