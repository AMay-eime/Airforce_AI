# ===== Original Version =====
# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]
#
# =====Modified Version =====
# Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)

# evaluation of policies or planning algorithms

from .agent import Agent


def view(env, player=None):
    if hasattr(env, 'view'):
        env.view(player=player)
    else:
        print(env)


def view_transition(env):
    if hasattr(env, 'view_transition'):
        env.view_transition()
    else:
        pass


def exec_match(env, models_for_policies, critic=None, show=False, game_args={}):
    ''' match with shared game environment '''
    if env.reset(game_args):
        return None
    agents = {}
    while not env.terminal():
        if show:
            view(env)
        if show and critic is not None:
            print('cv = ', critic.observe(env, None, show=False)[0])
        turn_players = env.turns()
        observers = env.observers()
        actions = {}
        for p in turn_players+observers:
            if(not p in agents):
                model=models_for_policies[env.policy_map[p]]
                deterministic = game_args['deterministics'][env.policy_map[p]]
                temperature = 0.0 if deterministic else 1.0
                agents[p]=Agent(model,temperature)
                agents[p].reset(env, show=show)
            agent=agents[p]
            if p in turn_players:
                actions[p] = agent.action(env, p, show=show)
            elif p in observers:
                agent.observe(env, p, show=show)
        if env.step(actions):
            return None
        if show:
            view_transition(env)
    outcome = env.outcome()
    if show:
        print('final outcome = %s' % outcome)
    return outcome


class Evaluator:
    def __init__(self, env, args):
        self.env = env
        self.args = args
        if(hasattr(env,"env")):
            #unwrap env
            self.match_monitor = args['custom_classes'][args["match_monitor_class"]](env.env)
        else:
            self.match_monitor = args['custom_classes'][args["match_monitor_class"]](env)

    def execute(self, models_for_policies, args):
        self.match_monitor.onEpisodeBegin()
        self.env.setMatch(args['match_info'])
        outcome = exec_match(self.env, models_for_policies,game_args=args)
        if outcome is None:
            print('None episode in evaluation!')
            return None
        args['player']=self.env.players()

        return {'args': args, 'score': self.env.score(),
            'policy_map': self.env.policy_map,
            'match_maker_result': self.match_monitor.onEpisodeEnd(args['match_type'])
        }
