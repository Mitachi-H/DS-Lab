# こっちが正しいそう

import numpy as np

class Izhikevich():
    def __init__(self, units: int, dt: float = 0.5):
        self.units = units
        self.dt = dt  # シミュレーションの時間刻み (ms)

        # Izhikevichモデルのパラメータ
        self.C = 100.0  # 膜容量 (pF)
        self.a = 0.03 * np.ones((units, 1))  # 回復時定数の逆数 (1/ms)
        self.b = -2.0 * np.ones((units, 1))  # 回復変数の感受性 (pA/mV)
        self.k = 0.7  # ゲイン (pA/mV)
        self.d = 100.0 * np.ones((units, 1))  # 発火で活性化される正味の外向き電流 (pA)
        self.vthr = -40.0  # 閾値電位 (mV)
        self.vrest = -60.0 * np.ones((units, 1))  # 静止膜電位 (mV)
        self.vreset = -50.0 * np.ones((units, 1))  # リセット電位 (mV)
        self.vpeak = 35.0  # ピーク電位 (mV)

        # 状態変数の初期化
        self.v = self.vrest.copy()  # 膜電位 (mV)
        self.u = self.b * (self.v - self.vrest)  # 回復変数

    def step(self, reservoir, x: np.ndarray) -> np.ndarray:
        """
        Izhikevichニューロンモデルのステップ関数。
        """
        u = x.reshape(-1, 1)  # 外部入力 (pA)
        # MEMO: 100倍しているのは、入力が1では小さすぎるため
        r = reservoir.state().T * 100  # スパイク入力 (pA)

        # リザーバーのパラメータ
        W = np.abs(reservoir.W)
        Win = np.abs(reservoir.Win)
        bias = reservoir.bias
        g_in = reservoir.noise_in
        g_rc = reservoir.noise_rc
        dist = reservoir.noise_type
        noise_gen = reservoir.noise_generator

        # 外部電流 I_ext の計算
        noise_in = noise_gen(dist=dist, shape=u.shape, gain=g_in)
        I_ext = W @ r + Win @ (u + noise_in) + bias

        # ノイズ項の追加
        noise_rc = noise_gen(dist=dist, shape=self.v.shape, gain=g_rc)

        # 膜電位と回復変数の更新 (Euler法)
        self.v += self.dt / self.C * (self.k * (self.v - self.vrest) * (self.v - self.vthr) - self.u + I_ext) + noise_rc
        self.u += self.dt * (self.a * (self.b * (self.v - self.vrest) - self.u))

        # スパイクの検出とリセット
        spikes = self.v >= self.vpeak
        self.v = np.where(spikes, self.vreset, self.v)
        self.u += np.where(spikes, self.d, 0)

        return spikes.astype(np.float32).T