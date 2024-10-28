# こっちが正しいそう

import numpy as np

class HHNeuron():
    def __init__(self, units: int, dt: float = 0.01):
        self.units = units
        self.dt = dt  # シミュレーションの時間刻み (s)

        # 生理学的定数
        self.C_m = 1.0  # 膜容量 (uF/cm^2)
        self.g_Na = 120.0  # Naの最大コンダクタンス (mS/cm^2)
        self.g_K = 36.0  # Kの最大コンダクタンス (mS/cm^2)
        self.g_L = 0.3  # 漏れチャネルのコンダクタンス (mS/cm^2)
        self.E_Na = 50.0  # Naの平衡電位 (mV)
        self.E_K = -77.0  # Kの平衡電位 (mV)
        self.E_L = -54.387  # 漏れチャネルの平衡電位 (mV)

        self.Vrest = -65.0  # 静止膜電位 (mV)

        # 状態変数の初期化
        self.v = self.Vrest * np.ones((units, 1))  # 膜電位 (mV)
        self.m = 0.05 * np.ones((units, 1))  # Naチャネルの活性化変数
        self.h = 0.6 * np.ones((units, 1))  # Naチャネルの不活性化変数
        self.n = 0.32 * np.ones((units, 1))  # Kチャネルの活性化変数

    # チャネルゲーティングの速度定数
    def alpha_m(self, v):
        V = v - self.Vrest
        return 0.1 * (25.0 - V) / (np.exp((25.0 - V) / 10.0) - 1)

    def beta_m(self, v):
        V = v - self.Vrest
        return 4.0 * np.exp(-V / 18.0)

    def alpha_h(self, v):
        V = v - self.Vrest
        return 0.07 * np.exp(-V / 20.0)

    def beta_h(self, v):
        V = v - self.Vrest
        return 1.0 / (1.0 + np.exp((30.0 - V) / 10.0))

    def alpha_n(self, v):
        V = v - self.Vrest
        return 0.01 * (10.0 - V) / (np.exp((10.0 - V) / 10.0) - 1)

    def beta_n(self, v):
        V = v - self.Vrest
        return 0.125 * np.exp(-V / 80.0)

    def step(self, reservoir, x: np.ndarray) -> np.ndarray:
        """
        Hodgkin-Huxleyニューロンモデルのステップ関数。
        """
        v_minus = self.v < 0

        u = x.reshape(-1, 1)  # 外部入力 (uA/cm^2)
        # MEMO: 30倍しているのは、入力が1では小さすぎるため
        r = reservoir.state().T * 10  # スパイク入力


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

        # Euler法による状態変数の更新
        alpha_m = self.alpha_m(self.v)
        beta_m = self.beta_m(self.v)
        alpha_h = self.alpha_h(self.v)
        beta_h = self.beta_h(self.v)
        alpha_n = self.alpha_n(self.v)
        beta_n = self.beta_n(self.v)

        # チャネルゲーティング変数の更新
        self.m += self.dt * (alpha_m * (1.0 - self.m) - beta_m * self.m)
        self.h += self.dt * (alpha_h * (1.0 - self.h) - beta_h * self.h)
        self.n += self.dt * (alpha_n * (1.0 - self.n) - beta_n * self.n)

        # イオン電流の計算
        I_Na = self.g_Na * self.m**3 * self.h * (self.v - self.E_Na)
        I_K = self.g_K * self.n**4 * (self.v - self.E_K)
        I_L = self.g_L * (self.v - self.E_L)

        noise_rc = noise_gen(dist=dist, shape=self.v.shape, gain=g_rc)
        # 膜電位の更新
        dV = self.dt * (I_ext - I_Na - I_K - I_L) / self.C_m + noise_rc
        self.v += dV

        # スパイクの検出
        spikes = (self.v >= 0) & v_minus

        return spikes.astype(np.float32).T
