# ===== Original Version =====
# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]
#
# =====Modified Version =====
# Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)

# worker and gather

import os
import random
import threading
import time
import functools
from socket import gethostname
from collections import defaultdict, deque, OrderedDict
import multiprocessing as mp
import cloudpickle
import copy
import torch

import queue
from .environment import prepare_env, make_env
from .connection import QueueCommunicator
from .connection import send_recv, open_multiprocessing_connections
from .connection import connect_socket_connection, accept_socket_connections
from .evaluation import Evaluator
from .generation import Generator
from .model import ModelWrapper
from ASRCAISim1.addons.HandyRLUtility.distribution import getActionDistributionClass


class Worker:
    def __init__(self, args, conn, wid):
        print('opened worker %d' % wid)
        self.worker_id = wid
        self.args = args
        self.exploration = None
        self.use_exploration = self.args.get("exploration_config",{}).get("use_exploration",False)
        if(self.use_exploration):
            self.exploration_cycle = 0
            self.exploration_config =  self.args["exploration_config"]
        self.conn = conn
        
        self.weight_pool = args['match_maker']["weight_pool"]

        self.env = make_env({**args['env'], 'id': wid})
        self.generator = Generator(self.env, self.args)
        self.evaluator = Evaluator(self.env, self.args)

        random.seed(args['seed'] + wid)
        self.numWeightsPerPolicy = 50
        self.custom_classes = self.args['custom_classes']
        self.model_pool = defaultdict(lambda:OrderedDict())
        self.model_epoch = -1
        for policyName,policyConfig in self.env.policy_config.items():
            model = self.get_model(policyName)
            initial_weight_path = policyConfig.get('initial_weight',None)
            if(initial_weight_path is not None):
                model.model.load_state_dict(torch.load(initial_weight_path, map_location=torch.device('cpu')), strict=True)
            model.eval()
            self.model_pool[policyName][0] = self.model_epoch, model

    def __del__(self):
        print('closed worker %d' % self.worker_id)

    def get_model(self,policyName):
        model_class = self.custom_classes[self.env.net(policyName)]
        obs_space = self.env.policy_config[policyName]['observation_space']
        ac_space = self.env.policy_config[policyName]['action_space']
        model_config = self.env.policy_config[policyName].get('model_config',{})
        if('actionDistributionClassGetter' in model_config):
            action_dist_class = self.custom_classes[model_config['actionDistributionClassGetter']](ac_space)
        else:
            action_dist_class = getActionDistributionClass(ac_space)
        return ModelWrapper(model_class(obs_space, ac_space, action_dist_class, model_config))

    def load_weight(self,policyName,weight_id):
        if(weight_id<=0):
            if(policyName == self.args['policy_to_train'] and self.model_epoch > self.model_pool[policyName][0][0]):
                model = ModelWrapper(cloudpickle.loads(send_recv(self.conn, ('model', (policyName,self.model_epoch)))))
                self.model_pool[policyName][0] = self.model_epoch, model
            model = self.model_pool[policyName][0][1]
        elif weight_id in self.model_pool[policyName]:
            model = self.model_pool[policyName].pop(weight_id)
            self.model_pool[policyName][weight_id] = model
        else:
            path=os.path.join(self.weight_pool,policyName+"-"+str(weight_id)+".pth")
            model = self.get_model(policyName)
            model.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')), strict=True)
            self.model_pool[policyName][weight_id] = model
            if(len(self.model_pool[policyName])>self.numWeightsPerPolicy):
                #メモリに保持している重みが上限を超えたら直近の読み込み時刻が最も古いもの(ただし、0を除く)を削除する。
                it=iter(self.model_pool[policyName].keys())
                removed=next(it)
                if(removed==0):
                    removed=next(it)
                self.model_pool[policyName].pop(removed,None)
        if model.training:
            model.eval()
        return model

    def _gather_models(self, match_info):
        models_for_policies = {}
        for team, info in match_info.items():
            models_for_policies[info["Policy"]+info["Suffix"]] = self.load_weight(info["Policy"], info["Weight"])
            if(info["Policy"] in self.args["policy_to_imitate"]):
                models_for_policies["Imitator"]=self.load_weight(self.args['policy_to_train'], -1)
        return models_for_policies

    def run(self):
        while True:
            args = send_recv(self.conn, ('args', None))
            if args is None:
                break
            role = args['role']
            self.model_epoch = args['model_epoch']
            models_for_policies = self._gather_models(args['match_info'])
            args['deterministics']={
                info["Policy"]+info["Suffix"]: info['deterministic']
                for info in args['match_info'].values()
            }

            if role == 'g':
                if(self.use_exploration):
                    alpha = self.exploration_config.get('alpha',0.7)
                    decay = self.exploration_config.get('eps_decay',-1)
                    if(decay <= 0):
                        eps = self.exploration_config.get('eps',self.exploration_config.get('eps_start',0.4))
                    else:
                        eps_start = self.exploration_config.get('eps_start',1.0)
                        eps_end = self.exploration_config.get('eps_end',0.1)
                        eps = eps_start + (eps_end - eps_start) * min(1.0,self.model_epoch / decay)
                    cycle = int(self.exploration_config.get('cycle',1))
                    self.exploration_cycle = (self.exploration_cycle + 1) % cycle
                    self.exploration = eps**(1+1.0*(self.worker_id*cycle+self.exploration_cycle)/(cycle*self.args['worker']['num_parallel']-1)*alpha)
                args['exploration'] = {
                    info["Policy"]+info["Suffix"]: self.exploration if info['Policy'] == self.args['policy_to_train'] and info['Weight'] < 0 else None
                    for info in args['match_info'].values()
                }
                episode = self.generator.execute(models_for_policies, args)
                send_recv(self.conn, ('episode', episode))
            elif role == 'e':
                result = self.evaluator.execute(models_for_policies, args)
                send_recv(self.conn, ('result', result))


def make_worker_args(args, n_ga, gaid, base_wid, wid, conn):
    return args, conn, base_wid + wid * n_ga + gaid


def open_worker(args, conn, wid):
    worker = Worker(args, conn, wid)
    worker.run()


class Gather(QueueCommunicator):
    def __init__(self, args, conn, gaid):
        print('started gather %d' % gaid)
        super().__init__()
        self.gather_id = gaid
        self.server_conn = conn
        self.args_queue = deque([])
        self.data_map = {'model': {}}
        self.result_send_map = {}
        self.result_send_cnt = 0

        n_pro, n_ga = args['worker']['num_parallel'], args['worker']['num_gathers']

        num_workers_per_gather = (n_pro // n_ga) + int(gaid < n_pro % n_ga)
        base_wid = args['worker'].get('base_worker_id', 0)

        worker_conns = open_multiprocessing_connections(
            num_workers_per_gather,
            open_worker,
            functools.partial(make_worker_args, args, n_ga, gaid, base_wid)
        )

        for conn in worker_conns:
            self.add_connection(conn)

        self.buffer_length = 1 + len(worker_conns) // 4

    def __del__(self):
        print('finished gather %d' % self.gather_id)

    def run(self):
        while self.connection_count() > 0:
            try:
                conn, (command, args) = self.recv(timeout=0.3)
            except queue.Empty:
                continue

            if command == 'args':
                # When requested arguments, return buffered outputs
                if len(self.args_queue) == 0:
                    # get multiple arguments from server and store them
                    self.server_conn.send((command, [None] * self.buffer_length))
                    self.args_queue += self.server_conn.recv()

                next_args = self.args_queue.popleft()
                self.send(conn, next_args)

            elif command in self.data_map:
                # answer data request as soon as possible
                data_id = args
                if data_id not in self.data_map[command]:
                    self.server_conn.send((command, args))
                    self.data_map[command][data_id] = self.server_conn.recv()
                self.send(conn, self.data_map[command][data_id])

            else:
                # return flag first and store data
                self.send(conn, None)
                if command not in self.result_send_map:
                    self.result_send_map[command] = []
                self.result_send_map[command].append(args)
                self.result_send_cnt += 1

                if self.result_send_cnt >= self.buffer_length:
                    # send datum to server after buffering certain number of datum
                    for command, args_list in self.result_send_map.items():
                        self.server_conn.send((command, args_list))
                        self.server_conn.recv()
                    self.result_send_map = {}
                    self.result_send_cnt = 0


def gather_loop(args, conn, gaid):
    gather = Gather(args, conn, gaid)
    gather.run()


class WorkerCluster(QueueCommunicator):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def run(self):
        # open local connections
        if 'num_gathers' not in self.args['worker']:
            self.args['worker']['num_gathers'] = 1 + max(0, self.args['worker']['num_parallel'] - 1) // 16
        for i in range(self.args['worker']['num_gathers']):
            conn0, conn1 = mp.Pipe(duplex=True)
            mp.Process(target=gather_loop, args=(self.args, conn1, i)).start()
            conn1.close()
            self.add_connection(conn0)


class WorkerServer(QueueCommunicator):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.total_worker_count = 0

    def run(self):
        # prepare listening connections
        def entry_server(port):
            print('started entry server %d' % port)
            conn_acceptor = accept_socket_connections(port=port)
            while True:
                conn = next(conn_acceptor)
                worker_args = conn.recv()
                print('accepted connection from %s!' % worker_args['address'])
                worker_args['base_worker_id'] = self.total_worker_count
                self.total_worker_count += worker_args['num_parallel']
                args = copy.deepcopy(self.args)
                args['worker'] = worker_args
                conn.send(args)
                conn.close()
            print('finished entry server')

        def worker_server(port):
            print('started worker server %d' % port)
            conn_acceptor = accept_socket_connections(port=port)
            while True:
                conn = next(conn_acceptor)
                self.add_connection(conn)
            print('finished worker server')

        threading.Thread(target=entry_server, args=(9999,), daemon=True).start()
        threading.Thread(target=worker_server, args=(9998,), daemon=True).start()


def entry(worker_args):
    conn = connect_socket_connection(worker_args['server_address'], 9999)
    conn.send(worker_args)
    args = conn.recv()
    conn.close()
    return args


class RemoteWorkerCluster:
    def __init__(self, args):
        args['address'] = gethostname()
        if 'num_gathers' not in args:
            args['num_gathers'] = 1 + max(0, args['num_parallel'] - 1) // 16

        self.args = args

    def run(self):
        args = entry(self.args)
        print(args)
        prepare_env(args['env'])

        # open worker
        process = []
        try:
            for i in range(self.args['num_gathers']):
                conn = connect_socket_connection(self.args['server_address'], 9998)
                p = mp.Process(target=gather_loop, args=(args, conn, i))
                p.start()
                conn.close()
                process.append(p)
            while True:
                time.sleep(100)
        finally:
            for p in process:
                p.terminate()


def worker_main(args, argv):
    # offline generation worker
    worker_args = args['worker_args']
    if len(argv) >= 1:
        worker_args['num_parallel'] = int(argv[0])

    worker = RemoteWorkerCluster(args=worker_args)
    worker.run()
