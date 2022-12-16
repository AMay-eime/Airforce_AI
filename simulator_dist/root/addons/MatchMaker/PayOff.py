# Estimated win rate class based on the pseudocode provided in the AlphaStar paper [1].
#
# [1] Vinyals, Oriol, et al. "Grandmaster level in StarCraft II using multi-agent reinforcement learning." Nature 575.7782 (2019): 350-354.
import numpy as np
from collections import defaultdict

class PayOff:
    """以下のAlphaStarの論文にて提供されている疑似コードに基づき実装した推定勝率のクラス。
    Vinyals, Oriol, et al. "Grandmaster level in StarCraft II using multi-agent reinforcement learning." Nature 575.7782 (2019): 350-354.
    """
    def __init__(self):
        self._games=defaultdict(lambda:0)
        self._wins=defaultdict(lambda:0)
        self._draws=defaultdict(lambda:0)
        self._losses=defaultdict(lambda:0)
        self._decay=0.99
    def _win_rate(self,home,away):
        if(self._games[home,away]==0):
            return 0.5
        return (self._wins[home,away]+0.5*self._draws[home,away])/self._games[home,away]
    def __getitem__(self,match):
        home,away=match
        if(isinstance(home,str)):
            home=[home]
        if(isinstance(away,str)):
            away=[away]
        win_rates=np.array([[self._win_rate(h,a) for a in away] for h in home])
        if(win_rates.shape[0]==1 or win_rates.shape[1]==1):
            win_rates=win_rates.reshape(-1)
        return win_rates
    def update(self,home,away,result):
        for stats in (self._games,self._wins,self._draws,self._losses):
            stats[home,away]*=self._decay
            stats[away,home]*=self._decay
        self._games[home,away]+=1
        self._games[away,home]+=1
        if(result=="win"):
            self._wins[home,away]+=1
            self._losses[away,home]+=1
        elif(result=="draw"):
            self._draws[home,away]+=1
            self._draws[away,home]+=1
        else:
            self._wins[away,home]+=1
            self._losses[home,away]+=1
