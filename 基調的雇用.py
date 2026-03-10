#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

import warnings
warnings.filterwarnings("ignore")

from fredapi import Fred
fred = Fred(api_key="")


# In[ ]:


data_list = {'PAYEMS':'Total',
             'USEHS':'Ed&Hlth',
             'USGOVT':'Govt',
             'USCONS':'Cons'}

for i, (k, v) in enumerate(data_list.items()):
    if i == 0:
        data = fred.get_series(k)
        data.name = v
    else:
        data = pd.concat([data,fred.get_series(k)],axis=1)
        data.columns = list(data.columns)[:-1]+[v]
        print(k)
data = data.dropna(how='any',axis=0)
data


# In[ ]:


df = data.copy()
df['core'] = df['Total'] - df['Ed&Hlth'] - df['Govt'] - df['Cons']
df['zero'] = 0
df.diff().loc['2022':,['zero','core']].plot()


# In[ ]:


data['Ed&Hlth'].diff().rolling(6).mean().loc['2025':].plot()


# In[ ]:


df.columns


# In[ ]:


df = df.iloc[:,:-1]
df.columns


# In[ ]:


df.tail()


# In[ ]:


adj = 31
df.loc['2026-02-01','Total'] += adj
df.loc['2026-02-01','Ed&Hlth'] += adj


# In[ ]:


df.tail()


# In[ ]:


# =========================
# グラフ準備
# =========================
import math
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from statsmodels.tsa.arima.model import ARIMA


# In[ ]:


# =========================
# 1. 1系列ごとの推計
# =========================
def fit_arima_series(y, col_name, h=12, p_range=range(4), q_range=range(4),
                     trend='n', diff_order=1):
    """
    y: 水準系列
    diff_order:
        1 -> 前月差 Δy を予測
        2 -> 前月差の差分 Δ²y を予測
    """
    y = y.copy()
    if y.index.freq is None:
        y = y.asfreq(pd.infer_freq(y.index) or "MS")
    
    # 表示用
    dy = y.diff().dropna()

    # 推計対象系列
    if diff_order == 1:
        target = dy.copy()
    elif diff_order == 2:
        target = dy.diff().dropna()
    else:
        raise ValueError("diff_order must be 1 or 2")

    target_model = target.copy()

    # Govt の 2025-10 レベルシフト対応
    # diff_order=1 なら 2025-10 の Δy が異常値
    # diff_order=2 なら 2025-10 と 2025-11 の Δ²y に影響しうる
    if col_name == "Govt":
        if diff_order == 1:
            target_model = target_model.drop("2025-10-01", errors="ignore")
        elif diff_order == 2:
            target_model = target_model.drop("2025-10-01", errors="ignore")
            target_model = target_model.drop("2025-11-01", errors="ignore")

    freq = pd.infer_freq(y.index) or "MS"
    target_model = target_model.asfreq(freq)
    
    best_aic = np.inf
    best_order = None
    best_model = None

    for p, q in product(p_range, q_range):
        try:
            model = ARIMA(target_model, order=(p, 0, q), trend=trend)
            res = model.fit(method_kwargs={"maxiter": 1000})

            if not res.mle_retvals.get("converged", False):
                continue

            if res.aic < best_aic:
                best_aic = res.aic
                best_order = (p, 0, q)
                best_model = res
        except Exception:
            continue

    if best_model is None:
        return None

    freq = pd.infer_freq(y.index) or "MS"
    future_index = pd.date_range(
        start=y.index[-1] + pd.tseries.frequencies.to_offset(freq),
        periods=h,
        freq=freq
    )

    # 予測
    forecast_target = best_model.forecast(steps=h)
    forecast_target.index = future_index

    # diff_order ごとに前月差へ戻す
    if diff_order == 1:
        forecast_dy = forecast_target.copy()
    else:
        # Δ²y を予測したので、最後の Δy から累積して Δy 予測へ戻す
        last_dy = dy.iloc[-1]
        forecast_dy = last_dy + forecast_target.cumsum()
        forecast_dy.index = future_index

    # 水準へ戻す
    forecast_level = y.iloc[-1] + forecast_dy.cumsum()
    forecast_level.index = future_index

    return {
        "y": y,                           # 水準実績
        "dy": dy,                         # 前月差実績
        "target": target,                 # 実際にモデル化した系列
        "target_model": target_model,     # 推計に使った系列
        "forecast_target": forecast_target,
        "forecast_dy": forecast_dy,       # 前月差予測
        "forecast_level": forecast_level, # 水準予測
        "order": best_order,
        "aic": best_aic,
        "diff_order": diff_order
    }

# =========================
# 2. グラフ描画
# =========================
def plot_result_set(results, kind="diff", ncols=2):
    n_series = len(results)
    nrows = math.ceil(n_series / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, (col, r) in enumerate(results.items()):
        ax = axes[i]

        if kind == "diff":
            ax.plot(r["dy"], label="Actual")
            ax.plot(r["forecast_dy"], label="Forecast")
            if not col == 'Total':
                ax.set_title(f"{col} Diff ARIMA{r['order']}")
            else:
                ax.set_title(f"{col}")
        elif kind == "level":
            ax.plot(r["y"], label="Actual")
            ax.plot(r["forecast_level"], label="Forecast")
            if not col == 'Total':
                ax.set_title(f"{col} Level ARIMA{r['order']}")
            else:
                ax.set_title(f"{col}") 
        else:
            raise ValueError("kind must be 'diff' or 'level'")

        ax.legend()

    # 余ったsubplotを削除
    for j in range(n_series, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()



# In[ ]:


def run_forecast_system(diff_order):
    # =========================
    # 1. 全系列を回して推計
    # =========================
    results = {}

    for col in df.columns:
        y = df.loc["2022":, col].copy()

        r = fit_arima_series(
            y,
            col_name=col,
            h=12,
            p_range=range(4),
            q_range=range(4),
            trend='n',
            diff_order=diff_order
        )

        if r is not None:
            results[col] = r
            print(f"{col}: ARIMA{r['order']}  AIC={r['aic']:.2f}")
        else:
            print(f"{col}: 推計失敗")

    # =========================
    # 2. Total を内訳の合計で作る
    # =========================
    components = ['Ed&Hlth', 'Govt', 'Cons', 'core']

    missing = [c for c in components if c not in results]
    if missing:
        raise ValueError(f"Missing component forecasts: {missing}")

    forecast_total_dy = sum(results[c]["forecast_dy"] for c in components)

    last_total = df.loc["2022":, "Total"].iloc[-1]
    forecast_total_level = last_total + forecast_total_dy.cumsum()

    if "Total" in results:
        results["Total"]["forecast_dy"] = forecast_total_dy
        results["Total"]["forecast_level"] = forecast_total_level
        results["Total"]["order"] = "sum"
    else:
        results["Total"] = {
            "y": df.loc["2022":, "Total"].copy(),
            "dy": df.loc["2022":, "Total"].copy().diff().dropna(),
            "forecast_dy": forecast_total_dy,
            "forecast_level": forecast_total_level,
            "order": "sum"
        }

    # =========================
    # 3. 差分グラフ一式
    # =========================
    plot_result_set(results, kind="diff", ncols=2)

    # =========================
    # 4. 水準グラフ一式
    # =========================
    plot_result_set(results, kind="level", ncols=2)

    return results


# In[ ]:


results1 = run_forecast_system(diff_order=1)
results2 = run_forecast_system(diff_order=2)


# In[ ]:


def combine_results(result_map, source_map, df, h=12, start_date="2022"):

    results_mix = {}

    for col, source in source_map.items():

        # -----------------
        # 1. ARIMA結果を使う
        # -----------------
        if isinstance(source, str):

            if source not in result_map:
                raise ValueError(f"Unknown source: {source}")

            results_mix[col] = result_map[source][col].copy()

        # -----------------
        # 2. 数値なら Δy を固定
        # -----------------
        elif isinstance(source, (int, float)):

            y = df.loc[start_date:, col].copy()
            dy = y.diff().dropna()

            freq = pd.infer_freq(y.index) or "MS"

            future_index = pd.date_range(
                start=y.index[-1] + pd.tseries.frequencies.to_offset(freq),
                periods=h,
                freq=freq
            )

            forecast_dy = pd.Series(source, index=future_index)

            forecast_level = y.iloc[-1] + forecast_dy.cumsum()

            results_mix[col] = {
                "y": y,
                "dy": dy,
                "forecast_dy": forecast_dy,
                "forecast_level": forecast_level,
                "order": f"fixed({source})"
            }

        else:
            raise ValueError(f"Unsupported source type for {col}")

    # -----------------
    # Total 再計算
    # -----------------
    components = list(source_map.keys())

    forecast_total_dy = sum(results_mix[c]["forecast_dy"] for c in components)

    last_total = df.loc[start_date:, "Total"].iloc[-1]
    forecast_total_level = last_total + forecast_total_dy.cumsum()

    results_mix["Total"] = {
        "y": df.loc[start_date:, "Total"].copy(),
        "dy": df.loc[start_date:, "Total"].copy().diff().dropna(),
        "forecast_dy": forecast_total_dy,
        "forecast_level": forecast_total_level,
        "order": "sum"
    }

    return results_mix


# In[ ]:


result_map = {
    "d1": results1,
    "d2": results2
}

source_map = {
    "Ed&Hlth": "d1",
    "Govt": 0,
    "Cons": "d1",
    "core": "d1"
}

results_mix = combine_results(result_map, source_map, df)
plot_result_set(results_mix, kind="diff", ncols=2)
plot_result_set(results_mix, kind="level", ncols=2)


# In[ ]:


pd.concat([
    pd.DataFrame(results_mix["Total"]["dy"]),
    pd.DataFrame(results_mix["Total"]["forecast_dy"].round(1))
]).loc['2025':]


# # ここまで

# In[ ]:





# In[ ]:


# =========================
# 1. 1系列ごとの推計: 次数制限
# =========================

def fit_arima_series3(y, col_name, h=12, diff_order=1):

    dy = y.diff().dropna()

    if diff_order == 1:
        target = dy.copy()
    elif diff_order == 2:
        target = dy.diff().dropna()
    else:
        raise ValueError("diff_order must be 1 or 2")

    target_model = target.copy()

    # Govt レベルシフト対応
    if col_name == "Govt":
        if diff_order == 1:
            target_model = target_model.drop("2025-10-01", errors="ignore")
        else:
            target_model = target_model.drop("2025-10-01", errors="ignore")
            target_model = target_model.drop("2025-11-01", errors="ignore")

    # 候補モデルを限定
    candidate_orders = [(1,0,0), (0,0,1), (1,0,1)]

    best_aic = np.inf
    best_order = None
    best_model = None

    for order in candidate_orders:

        try:
            model = ARIMA(target_model, order=order, trend='n')
            res = model.fit()

            if res.aic < best_aic:
                best_aic = res.aic
                best_order = order
                best_model = res

        except Exception:
            continue

    if best_model is None:
        return None

    freq = pd.infer_freq(y.index) or "MS"

    future_index = pd.date_range(
        start=y.index[-1] + pd.tseries.frequencies.to_offset(freq),
        periods=h,
        freq=freq
    )

    forecast_target = best_model.forecast(steps=h)
    forecast_target.index = future_index

    if diff_order == 1:

        forecast_dy = forecast_target.copy()

    else:

        last_dy = dy.iloc[-1]
        forecast_dy = last_dy + forecast_target.cumsum()
        forecast_dy.index = future_index

    forecast_level = y.iloc[-1] + forecast_dy.cumsum()
    forecast_level.index = future_index

    return {
        "y": y,
        "dy": dy,
        "target": target,
        "target_model": target_model,
        "forecast_target": forecast_target,
        "forecast_dy": forecast_dy,
        "forecast_level": forecast_level,
        "order": best_order,
        "aic": best_aic,
        "diff_order": diff_order,
        "model_type": "ARIMA",
        "result_obj": best_model
    }


# In[ ]:


# =========================
# 3. 全系列を回して推計
# =========================
results = {}

for col in df.columns:
    y = df.loc["2022":, col].copy()

    #r = fit_arima_series3(y, col_name=col, h=12, diff_order=1)
    r = fit_arima_series(y, col_name=col, h=12, p_range=range(4), q_range=range(4), trend='n', diff_order=1)
    #r = fit_ets_series(y, col_name=col, h=12, trend='n', diff_order=1)
    #r = fit_ucm_series(y, col_name=col, h=12, level_spec="local linear trend")

    if r is not None:
        results[col] = r
        print(f"{col}: ARIMA{r['order']}  AIC={r['aic']:.2f}")
    else:
        print(f"{col}: 推計失敗")

# =========================
# Total を内訳の合計で作る
# =========================
components = ['Ed&Hlth', 'Govt', 'Cons', 'core']

missing = [c for c in components if c not in results]
if missing:
    raise ValueError(f"Missing component forecasts: {missing}")

# 差分
forecast_total_dy = sum(results[c]["forecast_dy"] for c in components)

# 水準
last_total = df.loc["2022":, "Total"].iloc[-1]
forecast_total_level = last_total + forecast_total_dy.cumsum()

# Total が results にある場合は上書き、なければ新規作成
if "Total" in results:
    results["Total"]["forecast_dy"] = forecast_total_dy
    results["Total"]["forecast_level"] = forecast_total_level
else:
    results["Total"] = {
        "y": df.loc["2022":, "Total"].copy(),
        "dy": df.loc["2022":, "Total"].copy().diff().dropna(),
        "forecast_dy": forecast_total_dy,
        "forecast_level": forecast_total_level,
        "order": "sum"
    }

# =========================
# 4. 差分グラフ一式
# =========================
plot_result_set(results, kind="diff", ncols=2)

# =========================
# 5. 水準グラフ一式
# =========================
plot_result_set(results, kind="level", ncols=2)


# In[ ]:


# =========================
# 1. 1系列ごとの推計：　ETS
# =========================

def fit_ets_series(y, col_name, h=12, p_range=range(4), q_range=range(4),
                   trend='n', diff_order=1):

    y = y.astype(float).copy()
    dy = y.diff().dropna()

    target = y.copy()
    target_model = y.copy()

    # Govt の 2025-10 レベルシフト対応
    if col_name == "Govt":
        target_model = target_model.drop("2025-10-01", errors="ignore")

    try:
        model = ExponentialSmoothing(
            target_model,
            trend="add",
            damped_trend=True,
            seasonal=None,
            initialization_method="estimated"
        )
        res = model.fit(optimized=True)
    except Exception:
        return None

    freq = pd.infer_freq(y.index) or "MS"
    future_index = pd.date_range(
        start=y.index[-1] + pd.tseries.frequencies.to_offset(freq),
        periods=h,
        freq=freq
    )

    # 水準予測
    forecast_level = res.forecast(h)
    forecast_level = pd.Series(np.asarray(forecast_level), index=future_index)

    # 前月差予測
    forecast_dy = forecast_level.diff()
    forecast_dy.iloc[0] = forecast_level.iloc[0] - y.iloc[-1]

    return {
        "y": y,
        "dy": dy,
        "target": target,
        "target_model": target_model,
        "forecast_target": forecast_level,
        "forecast_dy": forecast_dy,
        "forecast_level": forecast_level,
        "order": "ETS(A,Ad,N)",
        "aic": getattr(res, "aic", np.nan),
        "diff_order": diff_order,
        "model_type": "ETS",
        "result_obj": res
    }


# In[ ]:


# =========================
# 1. 1系列ごとの推計：UCM
# =========================
def fit_ucm_series(y, col_name, h=12, level_spec="local linear trend"):
    """
    y: 水準系列（DatetimeIndex）
    col_name: 列名
    h: 予測期間
    level_spec: 'local level' または 'local linear trend'
    
    戻り値は、これまでの results[col] とほぼ同じ構造:
      - y
      - dy
      - target
      - target_model
      - forecast_target
      - forecast_dy
      - forecast_level
      - order
      - aic
      - diff_order
      - model_type
    """

    y = y.astype(float).copy()
    dy = y.diff().dropna()

    # Govt の 2025-10 レベルシフト対応:
    # 水準系列に pulse dummy を入れる
    exog = None
    future_exog = None

    if col_name == "Govt":
        exog = pd.DataFrame(0.0, index=y.index, columns=["pulse_2025_10"])
        if pd.Timestamp("2025-10-01") in exog.index:
            exog.loc["2025-10-01", "pulse_2025_10"] = 1.0

    try:
        model = UnobservedComponents(
            endog=y,
            level=level_spec,
            exog=exog
        )
        res = model.fit(disp=False)
    except Exception:
        return None

    freq = pd.infer_freq(y.index) or "MS"
    future_index = pd.date_range(
        start=y.index[-1] + pd.tseries.frequencies.to_offset(freq),
        periods=h,
        freq=freq
    )

    if exog is not None:
        future_exog = pd.DataFrame(0.0, index=future_index, columns=["pulse_2025_10"])
        forecast_level = pd.Series(
            res.forecast(steps=h, exog=future_exog),
            index=future_index
        )
    else:
        forecast_level = pd.Series(
            res.forecast(steps=h),
            index=future_index
        )

    # 前月差予測は、水準予測から作る
    level_full = pd.concat([y.iloc[[-1]], forecast_level])
    forecast_dy = level_full.diff().dropna()
    forecast_dy.index = future_index

    return {
        "y": y,                           # 水準実績
        "dy": dy,                         # 前月差実績
        "target": y,                      # UCMで直接モデル化した系列
        "target_model": y,                # 推計に使った系列
        "forecast_target": forecast_level,  # UCMでは forecast_target = 水準予測
        "forecast_dy": forecast_dy,       # 前月差予測
        "forecast_level": forecast_level, # 水準予測
        "order": level_spec,              # ARIMAの order の代わり
        "aic": res.aic,
        "diff_order": 0,                  # UCMは差分モデルではない
        "model_type": "UCM",
        "result_obj": res
    }

