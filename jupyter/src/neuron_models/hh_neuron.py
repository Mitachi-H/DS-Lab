import numpy as np
from scipy.integrate import solve_ivp

class HHNeuron():
    def __init__(self, units: int, dt: float = 0.01, method: str = 'Euler'):
        self.units = units
        self.dt = dt  # シミュレーションの時間刻み (s)
        self.method = method  # 数値解法の選択

        self.C_m = 1.0  # 膜容量 (uF/cm^2)
        self.g_Na = 120.0  # Naの最大コンダクタンス (mS/cm^2)
        self.g_K = 36.0
        self.g_L = 0.3
        self.E_Na = 50.0  # Naの平衡電位 (mV)
        self.E_K = -77.0
        self.E_L = -54.387

        self.Vrest = -65.0  # 静止膜電位 (mV)

        # 状態変数の初期化
        self.v = self.Vrest * np.ones((units, 1))  # 膜電位 (mV)
        self.m = 0.05 * np.ones((units, 1))
        self.h = 0.6 * np.ones((units, 1))
        self.n = 0.32 * np.ones((units, 1))

    # チャネルゲーティングの速度定数
    def alpha_m(self, V):
        V = np.clip(V, -100, 100)  # Vを-100から100の範囲に制限
        return 0.1 * (25.0 - V) / (np.exp((25.0 - V) / 10.0) - 1)

    def beta_m(self, V):
        V = np.clip(V, -100, 100)
        return 4.0 * np.exp(-V / 18.0)

    def alpha_h(self, V):
        V = np.clip(V, -100, 100)
        return 0.07 * np.exp(-V / 20.0)

    def beta_h(self, V):
        V = np.clip(V, -100, 100)
        return 1.0 / (1.0 + np.exp((30.0 - V) / 10.0))

    def alpha_n(self, V):
        V = np.clip(V, -100, 100)
        return 0.01 * (10.0 - V) / (np.exp((10.0 - V) / 10.0) - 1)

    def beta_n(self, V):
        V = np.clip(V, -100, 100)
        return 0.125 * np.exp(-V / 80.0)


    def step(self, reservoir, x: np.ndarray) -> np.ndarray:
        """
        Hodgkin-Huxleyニューロンモデルのステップ関数。
        """
        u = x.reshape(-1, 1)  # 外部入力 (uA/cm^2)
        r = reservoir.state().T  # スパイク入力

        W = np.abs(reservoir.W)
        Win = np.abs(reservoir.Win)
        bias = reservoir.bias
        g_in = reservoir.noise_in
        dist = reservoir.noise_type
        noise_gen = reservoir.noise_generator

        # 外部電流 I_ext の計算
        noise = noise_gen(dist=dist, shape=u.shape, gain=g_in)
        I_ext = W @ r + Win @ (u + noise) + bias  # 単位は uA/cm^2

        if self.method == 'Euler':
            # Euler法による更新
            alpha_m = self.alpha_m(self.v)
            beta_m = self.beta_m(self.v)
            alpha_h = self.alpha_h(self.v)
            beta_h = self.beta_h(self.v)
            alpha_n = self.alpha_n(self.v)
            beta_n = self.beta_n(self.v)

            self.m += self.dt * (alpha_m * (1.0 - self.m) - beta_m * self.m)
            self.h += self.dt * (alpha_h * (1.0 - self.h) - beta_h * self.h)
            self.n += self.dt * (alpha_n * (1.0 - self.n) - beta_n * self.n)

            # イオン電流の計算
            I_Na = self.g_Na * self.m**3 * self.h * (self.v - self.E_Na)
            I_K = self.g_K * self.n**4 * (self.v - self.E_K)
            I_L = self.g_L * (self.v - self.E_L)

            # 膜電位の更新
            dV = self.dt * (I_ext - I_Na - I_K - I_L) / self.C_m
            self.v += dV
        else:
            # ライブラリを使用した数値解法
            def hh_derivatives(t, y):
                units = self.units
                V = y[0:units]
                m = y[units:2*units]
                h = y[2*units:3*units]
                n = y[3*units:4*units]
                v = V - self.Vrest

                alpha_m = self.alpha_m(v)
                beta_m = self.beta_m(v)
                alpha_h = self.alpha_h(v)
                beta_h = self.beta_h(v)
                alpha_n = self.alpha_n(v)
                beta_n = self.beta_n(v)

                dVdt = (I_ext.flatten() - self.g_Na * m**3 * h * (V - self.E_Na)
                        - self.g_K * n**4 * (V - self.E_K)
                        - self.g_L * (V - self.E_L)) / self.C_m
                dmdt = alpha_m * (1 - m) - beta_m * m
                dhdt = alpha_h * (1 - h) - beta_h * h
                dndt = alpha_n * (1 - n) - beta_n * n

                dydt = np.concatenate([dVdt, dmdt, dhdt, dndt])
                return dydt

            # 初期条件の設定
            y0 = np.concatenate([self.v.flatten(), self.m.flatten(),
                                 self.h.flatten(), self.n.flatten()])

            # solve_ivpを使用した数値積分
            sol = solve_ivp(hh_derivatives, [0, self.dt], y0, method=self.method)

            # 状態変数の更新
            y_end = sol.y[:, -1]
            self.v = y_end[0:self.units].reshape(-1, 1)
            self.m = y_end[self.units:2*self.units].reshape(-1, 1)
            self.h = y_end[2*self.units:3*self.units].reshape(-1, 1)
            self.n = y_end[3*self.units:4*self.units].reshape(-1, 1)

        # スパイクの検出
        spikes = self.v >= 0

        return spikes.astype(np.float32).T
