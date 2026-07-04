"""
VT-LGARCH-t 模型：Volatility Targeting + Log-GARCH(EGARCH) + Student's t 殘差
標的：0050.TW（元大台灣50）

背景說明
--------
1. Log-GARCH / EGARCH：
   標準 GARCH 對 sigma^2 建模，需要參數為正才能保證變異數非負（常靠不等式約束處理）。
   EGARCH（Nelson, 1991）改對 log(sigma^2) 建模，天生滿足非負限制，不需要額外約束，
   且能捕捉「槓桿效應」（負報酬對波動的影響通常大於正報酬，透過不對稱項 gamma 捕捉）。
   本程式使用 EGARCH(1,1)：
       log(sigma_t^2) = omega + beta * log(sigma_{t-1}^2)
                        + alpha * (|z_{t-1}| - E|z_{t-1}|) + gamma * z_{t-1}
   其中 z_{t-1} = eps_{t-1} / sigma_{t-1} 為標準化殘差。

2. Volatility Targeting（波動率目標設定）：
   直接估計 omega 常與其他參數高度相關，導致估計不穩定、收斂慢。
   Volatility Targeting 的做法是「不直接估 omega」，而是先用樣本變異數（長期無條件變異數）
   反推 omega，讓模型的無條件變異數強制等於樣本變異數，藉此減少一個自由參數、加速並穩定收斂。
   對 EGARCH(1,1) 而言，無條件 log 變異數為 omega / (1 - beta)，
   因此若設定目標變異數 sigma^2_bar（樣本報酬變異數），則：
       omega = (1 - beta) * log(sigma^2_bar)
   在最佳化過程中，omega 不再是自由參數，而是由 beta 與 sigma^2_bar 反推出來（每次疊代重新計算）。

3. Student's t 殘差分佈：
   金融報酬率常有厚尾（extreme events 比常態分佈預期的更常出現）。
   假設標準化殘差 z_t ~ 標準化 Student's t 分佈（自由度 nu 為待估參數），
   能比常態假設更好地捕捉厚尾風險，nu 越小代表尾部越厚，nu 趨近無限大則退化為常態分佈。

套件安裝
--------
    pip install yfinance numpy pandas scipy matplotlib --break-system-packages

註：Python 的 `arch` 套件（arch_model）雖然支援 vol='EGARCH', dist='t'，
但「內建」的 vol_target 選項是針對標準 GARCH 設計，並不支援 EGARCH 的解析解變異數目標設定。
因此本程式改用 scipy.optimize 自行實作 EGARCH-t，並手動嵌入 Volatility Targeting 的重參數化技巧，
以完整達成「VT + Log-GARCH + t 分佈」三個要求。若只需要一般 GARCH(不含 VT)，
可另外改用 `arch` 套件的 `arch_model(returns, vol='EGARCH', p=1, o=1, q=1, dist='t')`。
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import optimize
from scipy.special import gammaln
from scipy.stats import t as student_t

# ------------------------------------------------------------------
# 1. 資料獲取：下載 0050.TW 歷史日收盤價，計算 Log Returns
# ------------------------------------------------------------------
TICKER = "0050.TW"
START = "2015-01-01"

def load_returns(ticker: str = TICKER, start: str = START) -> pd.Series:
    """下載收盤價並計算日對數報酬率（單位：%，避免數值過小造成最佳化不穩定）"""
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"下載失敗，請檢查代碼 {ticker} 或網路連線")
    close = df["Close"].dropna()
    # 對數報酬率 r_t = ln(P_t / P_{t-1})，乘以 100 轉換成百分比尺度，
    # 這是波動率模型的慣例做法，可讓概似函數數值更穩定、易於收斂。
    log_ret = np.log(close / close.shift(1)).dropna() * 100
    log_ret.name = "log_return"
    return log_ret.squeeze()


# ------------------------------------------------------------------
# 2. VT-EGARCH(1,1)-t 模型核心：負對數概似函數
# ------------------------------------------------------------------
class VT_EGARCH_t:
    """
    Volatility-Targeted EGARCH(1,1) with Student's t innovations.

    待估參數（不含 omega，omega 由 Volatility Targeting 反推）：
        beta  : log(sigma^2) 的持續性（AR(1) 係數），須落在 (-1, 1) 才平穩
        alpha : 對 |z| 衝擊的反應係數（波動幅度效應）
        gamma : 不對稱（槓桿）效應係數，gamma<0 代表負報酬讓波動放大更多
        nu    : Student's t 分佈的自由度（>2 才有有限變異數），越小尾部越厚
    """

    def __init__(self, returns: pd.Series):
        self.returns = returns.values.astype(float)
        self.n = len(self.returns)
        # 樣本變異數，作為 Volatility Targeting 的目標無條件變異數 sigma^2_bar
        self.target_var = float(np.var(self.returns, ddof=1))

    @staticmethod
    def _e_abs_z_t(nu: float) -> float:
        """
        標準化 Student's t 分佈下 E|z| 的解析解，
        用於 EGARCH 的 (|z_{t-1}| - E|z_{t-1}|) 去均值項，
        使該衝擊項的期望值為 0（模型設定的必要條件）。
        """
        # z 為「標準化」t 分布（變異數=1），標準 t(nu) 需除以 sqrt(nu/(nu-2)) 做尺度調整
        c = np.sqrt(nu / (nu - 2))
        # 標準 t(nu) 分布 |X| 的期望值解析式
        e_abs_std_t = 2 * np.sqrt(nu - 2) * np.exp(
            gammaln((nu + 1) / 2) - gammaln(nu / 2)
        ) / ((nu - 1) * np.sqrt(np.pi))
        return e_abs_std_t / c

    def _filter_variance(self, beta, alpha, gamma, nu):
        """遞迴計算條件變異數序列 sigma_t^2（EGARCH 遞迴 + Volatility Targeting 反推 omega）"""
        n = self.n
        r = self.returns
        log_h = np.empty(n)   # log(sigma_t^2)
        z = np.empty(n)

        # --- Volatility Targeting 核心：由 beta 與樣本變異數反推 omega ---
        # 無條件 log 變異數 = omega / (1 - beta)  =>  omega = (1 - beta) * log(sigma^2_bar)
        log_target_var = np.log(self.target_var)
        omega = (1 - beta) * log_target_var

        e_abs_z = self._e_abs_z_t(nu)

        # 初始值：以目標（無條件）變異數起始，是 Volatility Targeting 的自然設定
        log_h[0] = log_target_var
        h0 = np.exp(log_h[0])
        z[0] = r[0] / np.sqrt(h0)

        for tt in range(1, n):
            log_h[tt] = (
                omega
                + beta * log_h[tt - 1]
                + alpha * (np.abs(z[tt - 1]) - e_abs_z)
                + gamma * z[tt - 1]
            )
            # 數值保護：限制 log 變異數範圍，避免最佳化過程中偶發的參數組合造成溢位
            log_h[tt] = np.clip(log_h[tt], -20, 20)
            h_t = np.exp(log_h[tt])
            z[tt] = r[tt] / np.sqrt(h_t)

        h = np.exp(log_h)
        return h, omega

    def _neg_log_likelihood(self, params):
        beta, alpha, gamma, nu = params
        # 參數邊界的軟性懲罰（避免最佳化跑出可行域）
        if not (-0.999 < beta < 0.999) or nu <= 2.01:
            return 1e10

        h, _ = self._filter_variance(beta, alpha, gamma, nu)
        r = self.returns

        # 標準化 Student's t 分布的對數概似（變異數強制為 1，尺度用 sqrt(h_t) 承擔）
        c = np.sqrt(nu / (nu - 2))
        z = r / np.sqrt(h)
        # log f(r_t) = log t_std(z_t / c) - log(c) - 0.5*log(h_t)
        log_pdf_std_t = (
            gammaln((nu + 1) / 2)
            - gammaln(nu / 2)
            - 0.5 * np.log(np.pi * (nu - 2))
            - (nu + 1) / 2 * np.log(1 + (z / c) ** 2 / (nu - 2))
        )
        log_lik = log_pdf_std_t - np.log(c) - 0.5 * np.log(h)
        nll = -np.sum(log_lik)
        if not np.isfinite(nll):
            return 1e10
        return nll

    def fit(self):
        # 初始值：beta 給予高持續性、alpha/gamma 小幅擾動、nu 給一般厚尾常見值
        x0 = np.array([0.95, 0.10, -0.05, 8.0])
        bounds = [(-0.999, 0.999), (-1.0, 1.0), (-1.0, 1.0), (2.05, 50.0)]

        res = optimize.minimize(
            self._neg_log_likelihood,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-12},
        )
        self.result = res
        self.params_ = dict(zip(["beta", "alpha", "gamma", "nu"], res.x))

        # 以最終參數重新跑一次濾波，取得完整 h 序列與反推出的 omega
        h, omega = self._filter_variance(*res.x)
        self.params_["omega"] = omega
        self.h_ = h
        self.z_ = self.returns / np.sqrt(h)

        # 用數值 Hessian 近似標準誤（供 summary 使用）
        try:
            hess_inv = res.hess_inv.todense() if hasattr(res.hess_inv, "todense") else np.array(res.hess_inv)
            self.se_ = np.sqrt(np.diag(hess_inv))
        except Exception:
            self.se_ = np.full(4, np.nan)

        return self

    def summary(self):
        names = ["beta (持續性)", "alpha (衝擊反應)", "gamma (槓桿/不對稱)", "nu (t分布自由度)"]
        vals = [self.params_[k] for k in ["beta", "alpha", "gamma", "nu"]]
        lines = []
        lines.append("=" * 60)
        lines.append("VT-LGARCH(EGARCH)-t 模型估計結果")
        lines.append("=" * 60)
        lines.append(f"樣本數 n            : {self.n}")
        lines.append(f"目標無條件變異數(VT) : {self.target_var:.6f}  (= 樣本報酬變異數)")
        lines.append(f"反推得 omega        : {self.params_['omega']:.6f}  "
                      f"(= (1-beta) * log(目標變異數)，非自由估計參數)")
        lines.append("-" * 60)
        lines.append(f"{'參數':<22}{'估計值':>12}{'標準誤':>12}")
        for name, val, se in zip(names, vals, self.se_):
            lines.append(f"{name:<22}{val:>12.4f}{se:>12.4f}")
        lines.append("-" * 60)
        lines.append(f"Log-Likelihood      : {-self.result.fun:.4f}")
        lines.append(f"最佳化是否收斂      : {self.result.success}  ({self.result.message})")
        lines.append("=" * 60)
        return "\n".join(lines)

    def forecast(self, horizon: int = 5):
        """
        以最後估計狀態遞迴預測未來 horizon 天的條件波動率（年化與日化皆提供）。
        EGARCH 的多步預測需模擬未來衝擊項的期望值：
        由於 E[|z|] 與 E[z] (=0，t分布對稱) 已知，可直接用期望值遞迴（不需蒙地卡羅模擬）。
        """
        beta, alpha, gamma, nu = (self.params_[k] for k in ["beta", "alpha", "gamma", "nu"])
        omega = self.params_["omega"]
        e_abs_z = self._e_abs_z_t(nu)

        log_h_last = np.log(self.h_[-1])
        z_last = self.z_[-1]

        forecasts_h = []
        # 第一步用真實的最後一期 z_{T}
        log_h_next = omega + beta * log_h_last + alpha * (np.abs(z_last) - e_abs_z) + gamma * z_last
        forecasts_h.append(np.exp(log_h_next))

        # 第二步起，未來 z 的期望值：E[|z|]-E|z| 貢獻項期望為 0，E[z]=0（t分布對稱），
        # 故遞迴退化為 log_h_{t+k} = omega + beta * log_h_{t+k-1}
        log_h_prev = log_h_next
        for _ in range(1, horizon):
            log_h_prev = omega + beta * log_h_prev
            forecasts_h.append(np.exp(log_h_prev))

        forecasts_h = np.array(forecasts_h)
        daily_vol_pct = np.sqrt(forecasts_h)          # 日波動率（%，因報酬率已乘100）
        annualized_vol_pct = daily_vol_pct * np.sqrt(252)  # 年化波動率（假設一年252個交易日）

        return pd.DataFrame(
            {
                "day_ahead": np.arange(1, horizon + 1),
                "cond_variance_daily": forecasts_h,
                "cond_vol_daily_pct": daily_vol_pct,
                "cond_vol_annualized_pct": annualized_vol_pct,
            }
        )


# ------------------------------------------------------------------
# 3. 主程式
# ------------------------------------------------------------------
def main():
    print(f"下載 {TICKER} 資料並計算對數報酬率...")
    returns = load_returns()
    print(f"資料期間: {returns.index[0].date()} ~ {returns.index[-1].date()}，樣本數: {len(returns)}")

    print("\n擬合 VT-LGARCH(EGARCH)-t 模型...")
    model = VT_EGARCH_t(returns).fit()
    print(model.summary())

    print("\n未來 5 日條件波動率預測：")
    fc = model.forecast(horizon=5)
    print(fc.to_string(index=False))


if __name__ == "__main__":
    main()
