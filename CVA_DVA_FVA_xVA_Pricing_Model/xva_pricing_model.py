"""
CVA/DVA/FVA (xVA) Pricing Model
==================================
Prices CVA, DVA, and FVA for the same netting set as the Counterparty Credit Risk
Exposure Engine (2 IRS + 1 FX forward), using its real-SOFR-calibrated exposure profile,
a counterparty survival curve bootstrapped from a real market credit spread (FRED), and
the bank's own funding spread (real IG spread as a proxy) - then shows the collateral
mitigation effect of adding a CSA.
"""
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
import datetime

np.random.seed(7)
TODAY = datetime.date.today()

# ===========================================================================
# 1. Rebuild the real-calibrated exposure profile (same methodology as the
#    Counterparty Credit Risk Exposure Engine project, self-contained here)
# ===========================================================================
sofr = web.DataReader("SOFR", "fred", start=TODAY - datetime.timedelta(days=365 * 3)).dropna()
r = sofr.iloc[:, 0].values / 100
r_lag, r_now = r[:-1], r[1:]
b, a = np.polyfit(r_lag, r_now, 1)
dt = 1 / 252
kappa = -np.log(b) / dt if 0 < b < 1 else 0.5
theta = a / (1 - b) if b != 1 else r.mean()
sigma = (r_now - (a + b * r_lag)).std() * np.sqrt(1 / dt)
r0 = r[-1]
print(f"Real SOFR-calibrated Vasicek: kappa={kappa:.4f}, theta={theta:.4%}, "
      f"sigma={sigma:.4%}, r0={r0:.4%}")

fx = yf.download("EURUSD=X", period="1y", progress=False, auto_adjust=True)["Close"]
fx_series = fx["EURUSD=X"] if isinstance(fx, pd.DataFrame) else fx
fx_spot = float(fx_series.iloc[-1])
fx_vol = float(np.log(fx_series / fx_series.shift(1)).dropna().std() * np.sqrt(252))

N_PATHS, N_YEARS, STEPS_PER_YEAR = 5000, 5, 12
N_STEPS = N_YEARS * STEPS_PER_YEAR
dt_sim = 1 / STEPS_PER_YEAR

rates = np.zeros((N_PATHS, N_STEPS + 1)); rates[:, 0] = r0
fx_paths = np.zeros((N_PATHS, N_STEPS + 1)); fx_paths[:, 0] = fx_spot
for t in range(1, N_STEPS + 1):
    dW = np.random.normal(0, np.sqrt(dt_sim), N_PATHS)
    rates[:, t] = rates[:, t - 1] + kappa * (theta - rates[:, t - 1]) * dt_sim + sigma * dW
    dW_fx = np.random.normal(0, np.sqrt(dt_sim), N_PATHS)
    fx_paths[:, t] = fx_paths[:, t - 1] * np.exp(-0.5 * fx_vol**2 * dt_sim + fx_vol * dW_fx)

def swap_mtm(notional, fixed_rate, pay_fixed, short_rate, years_remaining):
    if years_remaining <= 0:
        return np.zeros_like(short_rate)
    duration = years_remaining * 0.9
    sign = 1 if pay_fixed else -1
    return sign * notional * duration * (fixed_rate - short_rate)

mtm = np.zeros((N_PATHS, N_STEPS + 1))
for t in range(N_STEPS + 1):
    ye = t * dt_sim
    mtm[:, t] += swap_mtm(50e6, 0.0380, True, rates[:, t], max(5 - ye, 0))
    mtm[:, t] += swap_mtm(30e6, 0.0410, False, rates[:, t], max(3 - ye, 0))
    if 2 - ye > 0:
        mtm[:, t] += 10e6 * (fx_paths[:, t] - fx_spot)

pos_exposure = np.maximum(mtm, 0)   # bank's exposure to counterparty default
neg_exposure = np.maximum(-mtm, 0)  # counterparty's exposure to the bank's default (for DVA)
epe = pos_exposure.mean(axis=0)
ene = neg_exposure.mean(axis=0)  # expected negative exposure (bank's own default risk side)
time_axis = np.arange(N_STEPS + 1) * dt_sim

# ===========================================================================
# 2. Bootstrap counterparty survival curve from a real market credit spread
# ===========================================================================
cpty_spread = web.DataReader("BAMLH0A0HYM2", "fred", start=TODAY - datetime.timedelta(days=30)).iloc[-1, 0] / 100
own_spread = web.DataReader("BAMLC0A0CM", "fred", start=TODAY - datetime.timedelta(days=30)).iloc[-1, 0] / 100
recovery_cpty = 0.40
recovery_own = 0.40
hazard_cpty = cpty_spread / (1 - recovery_cpty)
hazard_own = own_spread / (1 - recovery_own)
print(f"\nReal counterparty credit spread proxy (FRED HY index): {cpty_spread:.2%} "
      f"-> hazard rate {hazard_cpty:.2%}")
print(f"Real own (bank) credit spread proxy (FRED IG index): {own_spread:.2%} "
      f"-> hazard rate {hazard_own:.2%}")

survival_cpty = np.exp(-hazard_cpty * time_axis)
survival_own = np.exp(-hazard_own * time_axis)
default_prob_cpty = -np.diff(survival_cpty, prepend=1.0)
default_prob_own = -np.diff(survival_own, prepend=1.0)

# ===========================================================================
# 3. CVA, DVA (unilateral risk-neutral approximation, no CSA)
# ===========================================================================
cva = np.sum(epe[1:] * default_prob_cpty[1:] * (1 - recovery_cpty))
dva = np.sum(ene[1:] * default_prob_own[1:] * (1 - recovery_own))
print(f"\nCVA (no CSA) = sum[EPE(t) x PD(t) x LGD] = ${cva:,.0f}")
print(f"DVA (no CSA) = sum[ENE(t) x Own-PD(t) x Own-LGD] = ${dva:,.0f}")
print(f"Bilateral CVA (CVA - DVA) = ${cva - dva:,.0f}")

# ===========================================================================
# 4. FVA - funding cost/benefit of the uncollateralized position
# ===========================================================================
funding_spread = 0.0075  # 75bp bank funding spread over risk-free, typical unsecured funding cost
net_funding_exposure = epe - ene  # positive = bank funds the position, negative = bank benefits
fva = np.sum(net_funding_exposure[1:] * funding_spread * dt_sim)
print(f"\nFVA (funding cost at {funding_spread:.2%} spread over the exposure profile) = "
      f"${fva:,.0f}")
print(f"Total xVA adjustment (CVA - DVA + FVA) = ${cva - dva + fva:,.0f}")

# ===========================================================================
# 5. CSA effect: collateral threshold reduces EPE/ENE (reuse CSA logic)
# ===========================================================================
THRESHOLD = 2_000_000
epe_csa = np.minimum(epe, THRESHOLD)
ene_csa = np.minimum(ene, THRESHOLD)
cva_csa = np.sum(epe_csa[1:] * default_prob_cpty[1:] * (1 - recovery_cpty))
dva_csa = np.sum(ene_csa[1:] * default_prob_own[1:] * (1 - recovery_own))
fva_csa = np.sum((epe_csa - ene_csa)[1:] * funding_spread * dt_sim)

print("\n" + "=" * 70)
print("BEFORE vs. AFTER CSA ($2M threshold)")
print("=" * 70)
print(f"{'Metric':<10}{'No CSA':>15}{'With CSA':>15}{'Reduction':>15}")
for name, before, after in [("CVA", cva, cva_csa), ("DVA", dva, dva_csa), ("FVA", fva, fva_csa)]:
    reduction = 1 - after / before if before != 0 else 0
    print(f"{name:<10}${before:>13,.0f} ${after:>13,.0f} {reduction:>14.1%}")

total_before = cva - dva + fva
total_after = cva_csa - dva_csa + fva_csa
print(f"\nTotal xVA: ${total_before:,.0f} (no CSA) -> ${total_after:,.0f} (with CSA), "
      f"a {(1 - total_after/total_before):.1%} reduction")
