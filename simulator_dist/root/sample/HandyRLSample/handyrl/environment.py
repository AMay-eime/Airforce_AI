# ===== Original Version =====
# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]
#
# =====Modified Version =====
# Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)

# game environment

import importlib


ENVS = {
    'ASRCAISim1':         'handyrl.envs.ASRC.ASRCAISim1',
}


def prepare_env(env_args):
    env_name = env_args['env']
    env_source = ENVS.get(env_name, env_name)
    env_module = importlib.import_module(env_source)

    if env_module is None:
        print("No environment %s" % env_name)
    elif hasattr(env_module, 'prepare'):
        env_module.prepare()


def make_env(env_args):
    env_name = env_args['env']
    env_source = ENVS.get(env_name, env_name)
    env_module = importlib.import_module(env_source)

    if env_module is None:
        print("No environment %s" % env_name)
    else:
        return env_module.Environment(env_args)


# base class of Environment

class BaseEnvironment:
    def __init__(self, args={}):
        self.policy_map=[] # mapping from player id to policy name which should be defined in reset.
        self.policy_config = args.get("policy_config",{})
    def __str__(self):
        return ''

    #
    # Should be defined in all games
    #
    def reset(self, args={}):
        raise NotImplementedError()

    #
    # Should be defined in all games except you implement original step() function
    #
    def play(self, action, player):
        raise NotImplementedError()

    #
    # Should be defined in games which has simultaneous trainsition
    #
    def step(self, actions):
        for p, action in actions.items():
            if action is not None:
                self.play(action, p)

    #
    # Should be defined if you use multiplayer sequential action game
    #
    def turn(self):
        return 0

    #
    # Should be defined if you use multiplayer simultaneous action game
    #
    def turns(self):
        return [self.turn()]

    #
    # Should be defined if there are other players besides the turn player
    # who should observe the environment (mainly with RNNs)
    #
    def observers(self):
        return []

    #
    # Should be defined in all games
    #
    def terminal(self):
        raise NotImplementedError()

    #
    # Should be defined if you use immediate reward
    #
    def reward(self):
        return {}

    #
    # Should be defined in all games
    #
    def outcome(self):
        raise NotImplementedError()

    #
    # Score displayed during training. Score can be any value because it is not used for training. Default is outcome.
    #
    def score(self):
        return self.outcome()

    #
    # Should be defined in all games
    #
    def legal_actions(self, player):
        raise NotImplementedError()

    #
    # Should be defined in all games
    #
    def action_space(self, player):
        raise NotImplementedError()

    #
    # Should be defined if you use multiplayer game or add name to each player
    #
    def players(self):
        return [0]

    #
    # Should be defined in all games
    #
    def observation(self, player=None):
        raise NotImplementedError()

    #
    # Should be defined if you encode action as special string
    #
    def action2str(self, a, player=None):
        return str(a)

    #
    # Should be defined if you encode action as special string
    #
    def str2action(self, s, player=None):
        return int(s)

    #
    # Should be defined if you use network battle mode
    #
    def diff_info(self, player=None):
        return ''

    #
    # Should be defined if you use network battle mode
    #
    def update(self, info, reset):
        raise NotImplementedError()

    #
    # Shoule be defined if you use MatchMaker
    #
    def setMatch(self,matchInfo):
        self.policy_base={
            info["Policy"]+info["Suffix"]: info["Policy"]
            for info in matchInfo.values()
        }
        self.matchInfo=matchInfo

    #
    # Should be defined if you use imitation learning
    #
    def getImitationInfo(self):
        return {}
