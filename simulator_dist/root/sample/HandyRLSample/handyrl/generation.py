# ===== Original Version =====
# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]
#
# =====Modified Version =====
# Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)

# episode generation

import random
import bz2
import cloudpickle

import numpy as np
from gym import spaces

from .util import softmax, map_r

class Generator:
    def __init__(self, env, args):
        self.env = env
        self.args = args
        if(hasattr(env,"env")):
            #unwrap env
            self.match_monitor = args['custom_classes'][args["match_monitor_class"]](env.env)
        else:
            self.match_monitor = args['custom_classes'][args["match_monitor_class"]](env)
    def generate(self, models_for_policies, args):
        # episode generation
        moments = []
        models = {}
        hidden = {}
        deterministics={}
        exploration={}
        #replace config of env
        self.match_monitor.onEpisodeBegin()
        self.env.setMatch(args['match_info'])

        err = self.env.reset()
        if err:
            return None

        while not self.env.terminal():
            moment_keys = ['observation', 'selected_prob', 'legal_actions', 'action', 'value', 'reward', 'return', 'hidden_in']
            moment = {key: {p: None for p in self.env.players()} for key in moment_keys}

            turn_players = self.env.turns()
            observers = self.env.observers()
            for player in self.env.players():
                if player not in turn_players and (player not in observers or not self.args['observation']):
                    continue

                obs = self.env.observation(player)
                if not player in models:
                    models[player] = models_for_policies[self.env.policy_map[player]]
                    hidden[player] = models[player].init_hidden()
                    deterministics[player] = args['deterministics'][self.env.policy_map[player]]
                    exploration[player] = args['exploration'][self.env.policy_map[player]]

                model = models[player]
                moment['hidden_in'][player] = hidden[player]
                outputs = model.inference(obs, hidden[player])
                hidden[player] = outputs.get('hidden', None)
                v = outputs.get('value', None)

                moment['observation'][player] = obs
                moment['value'][player] = v

                if player in turn_players:
                    ac_space = self.env.action_space(player)
                    legal_actions_ = self.env.legal_actions(player)
                    dist = models[player].action_dist_class(ac_space, outputs['policy'], legal_actions_)
                    action_ = dist.sample(deterministics[player],exploration[player])
                    selected_prob_ = dist.log_prob(action_)
                    moment['selected_prob'][player] = selected_prob_
                    moment['legal_actions'][player] = legal_actions_
                    moment['action'][player] = action_
            err = self.env.step(moment['action'])
            if err:
                return None
            
            #教師役として用いる場合、模倣する側の情報を付加
            imitationInfo = self.env.getImitationInfo()
            for player,info in imitationInfo.items():
                if not player in models:
                    models[player] = models_for_policies[self.env.policy_map[player]] #"Imitator"
                    hidden[player] = models[player].init_hidden()
                    deterministics[player] = True
                model = models[player]
                outputs = model.inference(info['observation'], hidden[player])
                moment['hidden_in'][player] = hidden[player]
                moment['observation'][player] = info['observation']
                moment['action'][player] = info['action']
                moment['selected_prob'][player] = map_r(info['action'], lambda x: np.zeros_like(x,dtype=np.float32))
                moment['value'][player] = outputs.get('value', None)
                hidden[player] = outputs.get('hidden', None)
                ac_space = self.env.action_space(player)
                legal_actions_ = self.env.legal_actions(player)
                moment['legal_actions'][player] = legal_actions_

            reward = self.env.reward()
            for player in self.env.players():
                moment['reward'][player] = reward.get(player, None)

            moment['turn'] = turn_players + list(imitationInfo.keys())
            moments.append(moment)

        if len(moments) < 1:
            return None

        for player in self.env.players():
            ret = 0
            for i, m in reversed(list(enumerate(moments))):
                ret = (m['reward'][player] or 0) + self.args['gamma'] * ret
                moments[i]['return'][player] = ret
        args['player']=self.env.players()
        episode = {
            'args': args, 'steps': len(moments),
            'outcome': self.env.outcome(),
            'score': self.env.score(),
            'moment': [
                bz2.compress(cloudpickle.dumps(moments[i:i+self.args['compress_steps']]))
                for i in range(0, len(moments), self.args['compress_steps'])
            ],
            'policy_map': self.env.policy_map,
            'match_maker_result': self.match_monitor.onEpisodeEnd(args['match_type'])
        }

        return episode

    def execute(self, models, args):
        episode = self.generate(models, args)
        if episode is None:
            print('None episode in generation!')
        return episode
