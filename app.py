import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import optimize
from scipy.special import gammaln

# 設定中文字型（本機 Mac 用 PingFang，雲端環境用備用字型）
import os
_font_path = '/System/Library/AssetsV2/com_apple_MobileAsset_Font8/86ba2c91f017a3749571a82f2c6d890ac7ffb2fb.asset/AssetData/PingFang.ttc'
if os.path.exists(_font_path):
    fm.fontManager.addfont(_font_path)
    plt.rcParams['font.family'] = 'PingFang HK'
else:
    plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

class VT_EGARCH_t:
    def __init__(self, returns):
        self.returns = returns.values.astype(float)
        self.n = len(self.returns)
        self.target_var = float(np.var(self.returns, ddof=1))
    @staticmethod
    def _e_abs_z_t(nu):
        c = np.sqrt(nu / (nu - 2))
        e_abs_std_t = 2 * np.sqrt(nu - 2) * np.exp(gammaln((nu + 1) / 2) - gammaln(nu / 2)) / ((nu - 1) * np.sqrt(np.pi))
        return e_abs_std_t / c
    def _filter_variance(self, beta, alpha, gamma, nu):
        n, r = self.n, self.returns
        log_h, z = np.empty(n), np.empty(n)
        log_target_var = np.log(self.target_var)
        omega = (1 - beta) * log_target_var
        e_abs_z = self._e_abs_z_t(nu)
        log_h[0] = log_target_var
        z[0] = r[0] / np.sqrt(np.exp(log_h[0]))
        for tt in range(1, n):
            log_h[tt] = np.clip(omega + beta * log_h[tt-1] + alpha * (np.abs(z[tt-1]) - e_abs_z) + gamma * z[tt-1], -20, 20)
            z[tt] = r[tt] / np.sqrt(np.exp(log_h[tt]))
        return np.exp(log_h), omega
    def _neg_log_likelihood(self, params):
        beta, alpha, gamma, nu = params
        if not (-0.999 < beta < 0.999) or nu <= 2.01:
            return 1e10
        h, _ = self._filter_variance(beta, alpha, gamma, nu)
        c = np.sqrt(nu / (nu - 2))
        z = self.returns / np.sqrt(h)
        log_lik = (gammaln((nu+1)/2) - gammaln(nu/2) - 0.5*np.log(np.pi*(nu-2)) - (nu+1)/2*np.log(1+(z/c)**2/(nu-2))) - np.log(c) - 0.5*np.log(h)
        nll = -np.sum(log_lik)
        return nll if np.isfinite(nll) else 1e10
    def fit(self):
        res = optimize.minimize(self._neg_log_likelihood, [0.95, 0.10, -0.05, 8.0], method="L-BFGS-B", bounds=[(-0.999,0.999),(-1,1),(-1,1),(2.05,50)], options={"maxiter":2000,"ftol":1e-12})
        self.result = res
        self.params_ = dict(zip(["beta","alpha","gamma","nu"], res.x))
        h, omega = self._filter_variance(*res.x)
        self.params_["omega"] = omega
        self.h_ = h
        self.z_ = self.returns / np.sqrt(h)
        try:
            hess_inv = res.hess_inv.todense() if hasattr(res.hess_inv, "todense") else np.array(res.hess_inv)
            self.se_ = np.sqrt(np.diag(hess_inv))
        except:
            self.se_ = np.full(4, np.nan)
        return self
    def forecast(self, horizon=5):
        beta, alpha, gamma, nu = (self.params_[k] for k in ["beta","alpha","gamma","nu"])
        omega, e_abs_z = self.params_["omega"], self._e_abs_z_t(nu)
        log_h_last, z_last = np.log(self.h_[-1]), self.z_[-1]
        fh = []
        lh = omega + beta*log_h_last + alpha*(np.abs(z_last)-e_abs_z) + gamma*z_last
        fh.append(np.exp(lh))
        for _ in range(1, horizon):
            lh = omega + beta*lh
            fh.append(np.exp(lh))
        fh = np.array(fh)
        return pd.DataFrame({"天數": np.arange(1, horizon+1), "日波動率(%)": np.sqrt(fh).round(4), "年化波動率(%)": (np.sqrt(fh)*np.sqrt(252)).round(4)})

def interpret_results(ticker, params, h, fc):
    beta, alpha, gamma, nu = params['beta'], params['alpha'], params['gamma'], params['nu']
    vol_series = np.sqrt(h) * np.sqrt(252)
    current_vol = vol_series[-1]
    mean_vol = vol_series.mean()

    if beta > 0.97:
        beta_desc = f"**極高持續性（{beta:.4f}）**，波動率衝擊消散非常緩慢，市場情緒一旦改變將持續影響數週。"
    elif beta > 0.93:
        beta_desc = f"**高持續性（{beta:.4f}）**，波動率具有明顯慣性，今日的高波動往往延續至未來數日。"
    else:
        beta_desc = f"**中等持續性（{beta:.4f}）**，波動率會在衝擊後較快回歸正常水準。"

    if alpha > 0.15:
        alpha_desc = f"**較強衝擊反應（{alpha:.4f}）**，市場重大事件會顯著推升波動率。"
    elif alpha > 0.05:
        alpha_desc = f"**中等衝擊反應（{alpha:.4f}）**，市場衝擊對波動率有適度影響。"
    else:
        alpha_desc = f"**弱衝擊反應（{alpha:.4f}）**，波動率對單次衝擊不太敏感。"

    if gamma < -0.05:
        gamma_desc = f"**存在槓桿效應（{gamma:.4f}）**，股價下跌時波動率放大效果顯著大於上漲，反映投資人下跌時更為恐慌。"
    elif gamma > 0.05:
        gamma_desc = f"**反向槓桿效應（{gamma:.4f}）**，股價上漲時波動率反而放大，較為少見。"
    else:
        gamma_desc = f"**槓桿效應不明顯（{gamma:.4f}）**，漲跌對波動率的影響大致對稱。"

    if nu < 6:
        nu_desc = f"**厚尾特徵顯著（自由度 {nu:.1f}）**，極端報酬出現頻率遠高於常態分佈，需特別注意尾部風險。"
    elif nu < 15:
        nu_desc = f"**中等厚尾（自由度 {nu:.1f}）**，極端事件比常態分佈預期的略多，市場偶有黑天鵝事件。"
    else:
        nu_desc = f"**尾部接近常態（自由度 {nu:.1f}）**，{ticker} 作為大盤型標的，極端事件相對溫和，分散化效果良好。"

    vol_ratio = current_vol / mean_vol
    if vol_ratio > 1.3:
        vol_state = f"⚠️ **目前波動率偏高**（年化 {current_vol:.1f}%，高於歷史均值 {mean_vol:.1f}% 約 {(vol_ratio-1)*100:.0f}%），市場處於相對動盪狀態。"
    elif vol_ratio < 0.7:
        vol_state = f"😌 **目前波動率偏低**（年化 {current_vol:.1f}%，低於歷史均值 {mean_vol:.1f}% 約 {(1-vol_ratio)*100:.0f}%），市場相對平靜，但低波動往往孕育高波動。"
    else:
        vol_state = f"✅ **目前波動率接近正常水準**（年化 {current_vol:.1f}%，歷史均值 {mean_vol:.1f}%），市場狀態相對穩定。"

    trend = fc["年化波動率(%)"].iloc[-1] - fc["年化波動率(%)"].iloc[0]
    if trend < -0.5:
        forecast_desc = f"未來 {len(fc)} 日波動率預測**逐步下降**（{fc['年化波動率(%)'].iloc[0]:.1f}% → {fc['年化波動率(%)'].iloc[-1]:.1f}%），市場有望逐漸回穩。"
    elif trend > 0.5:
        forecast_desc = f"未來 {len(fc)} 日波動率預測**持續上升**（{fc['年化波動率(%)'].iloc[0]:.1f}% → {fc['年化波動率(%)'].iloc[-1]:.1f}%），建議留意風險。"
    else:
        forecast_desc = f"未來 {len(fc)} 日波動率預測**維持平穩**（約 {fc['年化波動率(%)'].mean():.1f}%），短期市場動能變化不大。"

    return beta_desc, alpha_desc, gamma_desc, nu_desc, vol_state, forecast_desc

st.set_page_config(page_title="VT-LGARCH-t", page_icon="📈", layout="wide")
st.title("📈 VT-LGARCH-t 波動率預測模型")
st.caption("Volatility Targeting + Log-GARCH(EGARCH) + Student's t 殘差")

with st.sidebar:
    st.header("⚙️ 設定")
    ticker = st.text_input("股票代號", value="0050.TW", help="台股加 .TW，例如 0050.TW、2330.TW、00878.TW")
    start_date = st.date_input("起始日", value=pd.Timestamp("2015-01-01"))
    horizon = st.slider("預測天數", 1, 30, 5)
    run_btn = st.button("🚀 執行模型", type="primary", use_container_width=True)
    st.divider()
    st.markdown("""
**模型說明**
- **EGARCH**：對 log(σ²) 建模，天生非負
- **Volatility Targeting**：固定無條件變異數
- **Student's t**：捕捉厚尾特徵
    """)

if run_btn:
    with st.spinner("下載資料並擬合模型..."):
        try:
            df = yf.download(ticker, start=str(start_date), auto_adjust=True, progress=False)
            if df.empty:
                st.error(f"找不到 {ticker}")
                st.stop()
            close = df["Close"].dropna().squeeze()
            returns = (np.log(close / close.shift(1)).dropna() * 100)
            returns.name = "log_return"
            model = VT_EGARCH_t(returns).fit()
            fc = model.forecast(horizon)
        except Exception as e:
            st.error(f"錯誤：{e}")
            st.stop()

    st.success(f"✅ 模型收斂！{returns.index[0].date()} ~ {returns.index[-1].date()}，樣本數 {len(returns)}")

    p = model.params_
    beta_desc, alpha_desc, gamma_desc, nu_desc, vol_state, forecast_desc = interpret_results(ticker, p, model.h_, fc)

    # 參數卡片（含 tooltip）
    st.subheader("📊 模型估計結果")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("beta（持續性）", f"{p['beta']:.4f}", help="波動率的自我相關係數。越接近1代表波動率慣性越強，衝擊消散越慢。台股通常在0.93~0.98之間。")
    c2.metric("alpha（衝擊反應）", f"{p['alpha']:.4f}", help="市場衝擊（大漲或大跌）對波動率的即時影響強度。數值越大代表波動率對新資訊反應越敏感。")
    c3.metric("gamma（槓桿效應）", f"{p['gamma']:.4f}", help="不對稱效應係數。負值代表下跌比上漲更會放大波動率（槓桿效應），符合多數股市的特性。")
    c4.metric("nu（t分布自由度）", f"{p['nu']:.2f}", help="Student's t分布的自由度。越小代表尾部越厚、極端事件越多。大於30時接近常態分布。")

    # 完整解讀報告
    with st.expander("📋 完整模型解讀報告", expanded=False):
        st.markdown(f"""
### {ticker} 模型解讀

---

#### 🔵 波動率持續性（beta）
{beta_desc}

> 當市場出現重大事件（如聯準會升息、地緣政治衝突），高 beta 意味著波動率將持續較長時間，投資人需要更長時間等待市場回穩。

---

#### 🟡 衝擊反應（alpha）
{alpha_desc}

> alpha 衡量「新衝擊」對波動率的立即效果。對指數型 ETF，alpha 通常比個股低，因為分散化已吸收了部分個股衝擊。

---

#### 🔴 槓桿效應（gamma）
{gamma_desc}

> 這是 EGARCH 相較標準 GARCH 最重要的優勢——能捕捉「恐慌不對稱性」。下跌 2% 對波動率的影響通常大於上漲 2%。

---

#### 🟣 尾部風險（nu，t分布自由度）
{nu_desc}

> 自由度越小，「黑天鵝事件」出現的機率越高。追蹤大盤的 ETF 自由度通常較高，反映分散化降低了極端尾部風險。

---

#### 📍 目前市場狀態
{vol_state}

#### 🔮 預測方向
{forecast_desc}

---
*Log-Likelihood：{-model.result.fun:.2f}　｜　omega（反推）：{p['omega']:.6f}*

> ⚠️ 以上解讀僅供參考，不構成投資建議。波動率預測描述的是風險環境，而非漲跌方向。
        """)

    st.divider()

    # 圖表
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📉 歷史條件波動率")
        fig, ax = plt.subplots(figsize=(7, 3.5))
        vs = pd.Series(np.sqrt(model.h_) * np.sqrt(252), index=returns.index)
        ax.plot(vs, color="#E74C3C", linewidth=0.8)
        ax.axhline(vs.mean(), color="gray", linestyle="--", label=f"均值 {vs.mean():.1f}%")
        ax.set_ylabel("年化波動率 (%)"); ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
        st.pyplot(fig)
    with col2:
        st.subheader("📊 日對數報酬率")
        fig2, ax2 = plt.subplots(figsize=(7, 3.5))
        ax2.plot(returns.index, returns.values, color="#3498DB", linewidth=0.6, alpha=0.7)
        ax2.axhline(0, color="gray"); ax2.set_ylabel("Log Return (%)"); ax2.grid(alpha=0.3); fig2.tight_layout()
        st.pyplot(fig2)

    st.divider()

    st.subheader(f"🔮 未來 {horizon} 日波動率預測")
    ca, cb = st.columns([1, 2])
    with ca:
        st.dataframe(fc, hide_index=True, use_container_width=True)
    with cb:
        fig3, ax3 = plt.subplots(figsize=(6, 3.5))
        ax3.bar(fc["天數"], fc["年化波動率(%)"], color="#2ECC71", alpha=0.8)
        for i, v in enumerate(fc["年化波動率(%)"].values):
            ax3.text(i+1, v+0.1, f"{v:.1f}%", ha="center", fontsize=9)
        ax3.set_xlabel("天數"); ax3.set_ylabel("年化波動率 (%)"); ax3.grid(alpha=0.3, axis="y"); fig3.tight_layout()
        st.pyplot(fig3)

    st.caption("⚠️ 以上預測僅供參考，不構成投資建議。")

else:
    st.info("👈 請在左側輸入股票代號，點擊執行模型開始分析。支援所有 Yahoo Finance 上的台股、美股代號。")