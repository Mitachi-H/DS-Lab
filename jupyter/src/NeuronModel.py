from abc import ABC, abstractmethod
import numpy as np
from scipy.sparse import issparse

class NeuronModel(ABC):
    @abstractmethod
    def step(self, reservoir, x: np.ndarray) -> np.ndarray:
        """
        ニューロンモデルのステップ関数。
        inputs: 外部からの入力, reservoir: リザーバの状態、重み行列や膜電位が含まれる
        出力: スパイクの有無を示す行列
        """
        pass

    def reservoir_kernel(self, reservoir, u, r):
        """Reservoir base forward function.

        Computes: s[t+1] = W.r[t] + Win.(u[t] + noise) + Wfb.(y[t] + noise) + bias
        """
        W = reservoir.W
        Win = reservoir.Win
        bias = reservoir.bias

        g_in = reservoir.noise_in
        dist = reservoir.noise_type
        noise_gen = reservoir.noise_generator

        pre_s = W @ r + Win @ (u + noise_gen(dist=dist, shape=u.shape, gain=g_in)) + bias

        if reservoir.has_feedback:
            Wfb = reservoir.Wfb
            g_fb = reservoir.noise_out
            h = reservoir.fb_activation

            y = reservoir.feedback().reshape(-1, 1)
            y = h(y) + noise_gen(dist=dist, shape=y.shape, gain=g_fb)

            pre_s += Wfb @ y

        return np.array(pre_s)

class LIFNeuronModel(NeuronModel):
    def __init__(self, time_constant=0.01, threshold=0.5, reset_potential=0.0):
        self.time_constant = time_constant
        self.threshold = threshold
        self.reset_potential = reset_potential

    def step(self, reservoir, x: np.ndarray) -> np.ndarray:
        """
        Leaky Integrate-and-Fire neuron model.

        """
        lr = reservoir.lr
        f = reservoir.activation
        dist = reservoir.noise_type
        g_rc = reservoir.noise_rc
        noise_gen = reservoir.noise_generator
        membrane_potentials = reservoir.membrane_potentials

        u = x.reshape(-1, 1)
        r = reservoir.state().T

        # 1. リセットされていないニューロンの膜電位を計算
        new_membrane_potentials = (
            (1 - lr) * membrane_potentials
            + lr * f(self.reservoir_kernel(reservoir, u, r))
            + noise_gen(dist=dist, shape=membrane_potentials.shape, gain=g_rc)
        )
        # print(f"new_membrane_potentials={new_membrane_potentials}")

        # 2. 閾値を超えたニューロンを検出
        spikes = new_membrane_potentials >= self.threshold

        # 3. 閾値を超えたニューロンの膜電位をリセット
        new_membrane_potentials[spikes] = self.reset_potential

        # 膜電位の更新
        reservoir.membrane_potentials = new_membrane_potentials

        return spikes.astype(np.float32).T

