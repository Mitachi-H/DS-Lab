import numpy as np

class LIFNeuron():
    def __init__(self, units: int):
        self.membrane_potentials = np.zeros((units, 1))
        self.refractory_timer = np.zeros((units, 1))

        self.tref = 2  # 不応期(ms = step)
        self.tc_m = 10  # 膜時定数(ms)
        self.vrest = -60  # 静止膜電位(mV)
        self.vreset = -65  # リセット電位(mV)
        self.vthr = -50  # 閾値電位(mV)
        self.vpeak = 30  # ピーク電位(mV)
        self.R_m = 10  # 膜抵抗(MΩ)

    def step(self, reservoir, x: np.ndarray) -> np.ndarray:
        """
        Leaky Integrate-and-Fire neuron model with refractory period and peak potential.
        """
        u = x.reshape(-1, 1)  # 外部入力 (nA)
        r = reservoir.state().T  # スパイク入力

        W = np.abs(reservoir.W)
        Win = np.abs(reservoir.Win)
        bias = reservoir.bias
        g_in = reservoir.noise_in
        dist = reservoir.noise_type
        noise_gen = reservoir.noise_generator

        I_ext = np.array(
            W @ r + Win @ (u + noise_gen(dist=dist, shape=u.shape, gain=g_in)) + bias
        )

        # 不応期タイマーを減算
        self.refractory_timer = np.maximum(self.refractory_timer - 1, 0)

        # 不応期ではないニューロンのインデックスを取得
        not_refractory = self.refractory_timer == 0

        # 不応期ではないニューロンの膜電位を更新
        self.membrane_potentials[not_refractory] += (
            (self.vrest - self.membrane_potentials[not_refractory] + self.R_m * I_ext[not_refractory])
            / self.tc_m
        )

        # 不応期中のニューロンの膜電位をリセット電位に設定
        self.membrane_potentials[self.refractory_timer > 0] = self.vreset

        # 閾値を超えたニューロンを検出
        spikes = self.membrane_potentials >= self.vthr

        # スパイクしたニューロンの膜電位をピーク電位に設定し、不応期タイマーを設定
        self.membrane_potentials[spikes] = self.vpeak
        self.refractory_timer[spikes] = self.tref

        return spikes.astype(np.float32).T
