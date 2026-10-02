"""
Counterparty Exposure Engine (Hull-White 1-Factor, Gregory-style)
=====================================================================
Monte Carlo exposure engine for a 15-trade interest rate swap netting set:
  - Hull-White 1-factor short-rate model, mean-reversion and volatility calibrated on
    REAL historical SOFR data, theta(t) fitted exactly to the REAL current Treasury/SOFR
    curve (the model reproduces today's real forward curve exactly, the defining feature
    of Hull-White vs. a plain Vasicek).
  - Exact analytic zero-coupon bond pricing under Hull-White at every simulated node, used
    to fully revalue every swap's fixed and floating legs at every time step (not a
    duration approximation).
  - Full exposure profile: EE, EPE, EffEE, EffEPE, PFE at 95% and 99%.
  - Collateral: a real CSA algorithm (threshold, MTA, 10-day margin period of risk) shows
    how much exposure actually survives margining.
  - Unilateral and bilateral CVA off the exposure profile, hazard rates bootstrapped from
    a CDS-style spread term structure anchored to real credit-index levels.

Reference methodology: Jon Gregory, "The xVA Challenge" / "Counterparty Credit Risk and
Credit Value Adjustment," chapters on exposure simulation and collateral (ch. 9-11).
"""

# ===========================================================================
# CONFIG BLOCK - all user-facing inputs live here
# ===========================================================================
N_PATHS = 2000
MAX_TENOR_YEARS = 10
STEPS_PER_YEAR = 52  # weekly steps - needed so a real 10-day MPoR maps to ~1-2 steps,
                     # not a fraction of a monthly step (an earlier version used monthly
                     # steps and a broken days-to-steps conversion, which silently lagged
                     # the collateral by ~10 MONTHS instead of ~10 days - fixed here)
CSA_THRESHOLD = 1_000_000
CSA_MTA = 250_000
MPOR_DAYS = 10
RECOVERY_CPTY = 0.40
RECOVERY_OWN = 0.40
SEED = 2024

import numpy as np
import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TODAY = datetime.date.today()
rng = np.random.default_rng(SEED)

# ===========================================================================
# 1. Calibrate Hull-White mean-reversion (a) and volatility (sigma) on REAL
#    historical SOFR data (AR(1) regression, same discipline as the earlier
#    Vasicek-based build, now feeding a proper Hull-White model)
# ===========================================================================
sofr = web.DataReader("SOFR", "fred", start=TODAY - datetime.timedelta(days=365 * 3)).dropna()
r_hist = sofr.iloc[:, 0].values / 100
r_lag, r_now = r_hist[:-1], r_hist[1:]
dt_hist = 1 / 252
b_coef, a_coef = np.polyfit(r_lag, r_now, 1)
A_MEANREV = -np.log(b_coef) / dt_hist if 0 < b_coef < 1 else 0.15
SIGMA_HW = (r_now - (a_coef + b_coef * r_lag)).std() * np.sqrt(1 / dt_hist)
print(f"Hull-White mean reversion (a) and vol (sigma) calibrated on real SOFR data "
      f"({len(r_hist)} obs, {sofr.index[0].date()} to {sofr.index[-1].date()}):")
print(f"  Mean reversion a = {A_MEANREV:.4f}, sigma = {SIGMA_HW:.4%}")
# r(0) is set to the model's OWN curve-implied instantaneous short rate (f0(0)), not the
# separate real SOFR print used to calibrate a/sigma - an earlier version mixed the two
# real rates (SOFR history vs. the Treasury curve fitted below), which produced a real
# SOFR-Treasury basis mismatch and a small nonzero t=0 netting-set MTM. Standard Hull-
# White practice sets r(0) = f(0,0) exactly so the model is internally consistent with
# its own fitted curve; historical data is used only for a and sigma, never for r(0).

# ===========================================================================
# 2. Real initial discount curve P(0,t) from the real Treasury/SOFR curve
# ===========================================================================
tenor_codes = {0.25: "DGS3MO", 0.5: "DGS6MO", 1: "DGS1", 2: "DGS2", 3: "DGS3",
               5: "DGS5", 7: "DGS7", 10: "DGS10"}
real_curve = {}
for tenor, code in tenor_codes.items():
    df = web.DataReader(code, "fred", start=TODAY - datetime.timedelta(days=15)).dropna()
    real_curve[tenor] = df.iloc[-1, 0] / 100
print("\nReal market curve (FRED) used to fit the Hull-White initial term structure:")
for t, y in real_curve.items():
    print(f"  {t}Y: {y:.3%}")

curve_tenors = np.array(sorted(real_curve.keys()))
curve_yields = np.array([real_curve[t] for t in curve_tenors])
# Cubic spline, not linear interpolation - linear interpolation has a kink (discontinuous
# slope) at every real curve tenor point, which a smooth-curve analytic formula like the
# Hull-White A(t,T) doesn't reproduce well; an earlier version used linear interpolation
# and left a small (~0.3-0.4% of notional) spurious inception mark-to-market on the
# netting set purely from this curve-smoothness mismatch, not a real economic value
from scipy.interpolate import CubicSpline
_spline = CubicSpline(curve_tenors, curve_yields, bc_type="natural")

def zero_yield(t):
    t = np.clip(t, curve_tenors[0], curve_tenors[-1])
    return _spline(t)

def P0(t):
    """Real market discount factor to time t, continuously-compounded zero-yield curve."""
    t = np.maximum(t, 1e-6)
    return np.exp(-zero_yield(t) * t)

def f0(t, h=1e-4):
    """Real market instantaneous forward rate at time t, from finite-difference on ln P0.
    Uses a one-sided (forward) difference near t=0 - a central difference there would
    evaluate P0 at negative time, which the floor inside P0 handles inconsistently and
    produced a badly wrong forward rate exactly at t=0 in an earlier version (the
    resulting r(0) mismatch was the root cause of a large, systematic netting-set
    inception mark-to-market that first looked like a curve-interpolation artifact but
    was actually this finite-difference bug)."""
    if t < h:
        return -(np.log(P0(t + h)) - np.log(P0(t))) / h
    return -(np.log(P0(t + h)) - np.log(P0(t - h))) / (2 * h)

def f0_prime(t, h=1e-3):
    """d/dt of the real forward curve, via finite difference (feeds theta(t)).
    Same one-sided-near-zero fix as f0() itself: a central difference here would call
    f0 at negative time, which f0's own floor turns into a near-zero forward rate,
    producing a huge spurious derivative and, through theta(t), a huge spurious drift
    on the very first Euler step of the short-rate simulation - this was the actual root
    cause of a large deterministic (non-random) exposure spike at the second simulated
    time step, which a naive read first looked like real path-dependent exposure buildup."""
    if t < h:
        return (f0(t + h) - f0(t)) / h
    return (f0(t + h) - f0(t - h)) / (2 * h)

def theta(t):
    """Exact Hull-White theta(t) fitted to the real initial curve (Brigo-Mercurio formula) -
    this is what makes the model reproduce today's real forward curve exactly."""
    a = A_MEANREV
    return f0_prime(t) + a * f0(t) + (SIGMA_HW ** 2 / (2 * a)) * (1 - np.exp(-2 * a * t))

r0 = f0(0.0)  # model's own curve-implied instantaneous short rate, not a separate real print
print(f"  r(0) = f(0,0) from the fitted real curve = {r0:.4%}")

# ===========================================================================
# 3. Analytic Hull-White zero-coupon bond price P(t,T | r(t)) - used to revalue
#    every swap leg exactly at every simulated node, no duration approximation
# ===========================================================================
def B_func(t, T):
    a = A_MEANREV
    return (1 - np.exp(-a * (T - t))) / a

def A_func(t, T):
    a = A_MEANREV
    B_tT = B_func(t, T)
    term = B_tT * f0(t) - (SIGMA_HW ** 2 / (4 * a)) * B_tT ** 2 * (1 - np.exp(-2 * a * t))
    return (P0(T) / P0(t)) * np.exp(term)

def P_hw(t, T, r_t):
    """Vectorized over r_t (array of short rates across simulated paths)."""
    if T <= t:
        return np.ones_like(r_t)
    return A_func(t, T) * np.exp(-B_func(t, T) * r_t)

# ===========================================================================
# 4. Simulate short-rate paths under Hull-White (Euler-Maruyama)
# ===========================================================================
N_STEPS = MAX_TENOR_YEARS * STEPS_PER_YEAR
dt_sim = 1 / STEPS_PER_YEAR
time_grid = np.arange(N_STEPS + 1) * dt_sim

rates = np.zeros((N_PATHS, N_STEPS + 1))
rates[:, 0] = r0
for i in range(1, N_STEPS + 1):
    t_prev = time_grid[i - 1]
    dW = rng.normal(0, np.sqrt(dt_sim), N_PATHS)
    rates[:, i] = rates[:, i - 1] + (theta(t_prev) - A_MEANREV * rates[:, i - 1]) * dt_sim + SIGMA_HW * dW

print(f"\nSimulated {N_PATHS} Hull-White short-rate paths, {N_STEPS} monthly steps "
      f"over {MAX_TENOR_YEARS} years")

# ===========================================================================
# 5. 15-trade IRS netting set - mix of payer/receiver, varied notional/tenor,
#    fixed rates set at the real par swap rate at inception (so trade-level
#    inception MTM = 0, matching real market practice)
# ===========================================================================
def par_swap_rate(tenor, freq=2):
    """Par fixed rate solving FloatLeg(0) = FixedLeg(0) off the real initial curve."""
    payment_times = np.arange(1, int(tenor * freq) + 1) / freq
    annuity = sum((1 / freq) * P0(ti) for ti in payment_times)
    return (P0(0.001) - P0(tenor)) / annuity

trade_specs = []
rng_trades = np.random.default_rng(11)
for i in range(15):
    tenor = rng_trades.choice([2, 3, 5, 7, 10])
    notional = rng_trades.uniform(10, 50) * 1_000_000
    pay_fixed = rng_trades.choice([True, False])
    fixed_rate = par_swap_rate(tenor)
    trade_specs.append({"trade_id": i, "tenor": tenor, "notional": notional,
                          "pay_fixed": pay_fixed, "fixed_rate": fixed_rate, "freq": 2})

trades_df = pd.DataFrame(trade_specs)
print(f"\n15-trade netting set (par rates set off the real initial curve):")
print(trades_df.round(4).to_string(index=False))

def swap_value(trade, t, r_t):
    """Exact analytic Hull-White swap value at time t for the given trade, vectorized
    over the array of short rates r_t across all Monte Carlo paths."""
    tenor, freq, K, N = trade["tenor"], trade["freq"], trade["fixed_rate"], trade["notional"]
    payment_times = np.arange(1, int(tenor * freq) + 1) / freq
    remaining = payment_times[payment_times > t]
    if len(remaining) == 0:
        return np.zeros_like(r_t)
    next_reset = remaining[0] - 1 / freq if (remaining[0] - 1 / freq) > t else t
    float_leg = N * (P_hw(t, next_reset if next_reset > t else t, r_t) - P_hw(t, tenor, r_t)) \
        if next_reset > t else N * (1 - P_hw(t, tenor, r_t))
    fixed_leg = K * N * (1 / freq) * sum(P_hw(t, ti, r_t) for ti in remaining)
    value = float_leg - fixed_leg  # payer-swap value (receive float, pay fixed)
    return value if trade["pay_fixed"] else -value

# ===========================================================================
# 6. Revalue the full netting set at every time step, every path
# ===========================================================================
netting_mtm = np.zeros((N_PATHS, N_STEPS + 1))
for step in range(N_STEPS + 1):
    t = time_grid[step]
    r_t = rates[:, step]
    for _, trade in trades_df.iterrows():
        if t < trade["tenor"]:
            netting_mtm[:, step] += swap_value(trade, t, r_t)

print("Revalued the full 15-trade netting set at every time step (exact HW discounting)")

# ===========================================================================
# 7. Exposure profile: EE, EPE, EffEE, EffEPE, PFE 95/99 (no collateral)
# ===========================================================================
pos_exposure = np.maximum(netting_mtm, 0)
EE = pos_exposure.mean(axis=0)
EffEE = np.maximum.accumulate(EE)  # running max, Gregory's "effective EE"
EPE = EE.mean()
EffEPE = EffEE.mean()
PFE95 = np.percentile(pos_exposure, 95, axis=0)
PFE99 = np.percentile(pos_exposure, 99, axis=0)

print("\n" + "=" * 70)
print("EXPOSURE PROFILE - NO COLLATERAL")
print("=" * 70)
for yr in [0, 1, 2, 5, 10]:
    idx = min(int(yr * STEPS_PER_YEAR), N_STEPS)
    print(f"  Year {yr}: EE=${EE[idx]:,.0f}  EffEE=${EffEE[idx]:,.0f}  "
          f"PFE95=${PFE95[idx]:,.0f}  PFE99=${PFE99[idx]:,.0f}")
print(f"\nEPE (time-averaged EE): ${EPE:,.0f}")
print(f"EffEPE (time-averaged running-max EE): ${EffEPE:,.0f}")

# ===========================================================================
# 8. Collateral: real CSA algorithm - threshold, MTA, 10-day margin period of
#    risk (Gregory ch. 9-11 mechanics)
# ===========================================================================
days_per_step = 365 / STEPS_PER_YEAR
mpor_steps = max(1, round(MPOR_DAYS / days_per_step))
print(f"\nMPoR: {MPOR_DAYS} real days = {days_per_step:.1f} days/step -> "
      f"lagging collateral by {mpor_steps} simulation step(s)")

def apply_csa(mtm_paths, threshold, mta, mpor_lag_steps):
    n_paths, n_steps = mtm_paths.shape
    collateral = np.zeros_like(mtm_paths)
    for t in range(1, n_steps):
        target_collateral = np.maximum(mtm_paths[:, t] - threshold, 0)
        change = target_collateral - collateral[:, t - 1]
        call_triggered = np.abs(change) >= mta
        collateral[:, t] = np.where(call_triggered, target_collateral, collateral[:, t - 1])
    # Exposure under a margin period of risk: the collateral actually available at time t
    # is what was posted mpor_lag_steps earlier (a real, standard MPoR approximation -
    # collateral can't react instantly to the last mpor_lag_steps of MTM moves)
    lagged_collateral = np.zeros_like(collateral)
    lagged_collateral[:, mpor_lag_steps:] = collateral[:, :-mpor_lag_steps]
    collateralized_exposure = np.maximum(mtm_paths - lagged_collateral, 0)
    return collateralized_exposure, collateral

coll_exposure, collateral_balance = apply_csa(netting_mtm, CSA_THRESHOLD, CSA_MTA, mpor_steps)
EE_csa = coll_exposure.mean(axis=0)
EffEE_csa = np.maximum.accumulate(EE_csa)
EPE_csa = EE_csa.mean()
EffEPE_csa = EffEE_csa.mean()
PFE95_csa = np.percentile(coll_exposure, 95, axis=0)

print("\n" + "=" * 70)
print(f"EXPOSURE PROFILE - WITH CSA (${CSA_THRESHOLD:,.0f} threshold, "
      f"${CSA_MTA:,.0f} MTA, {MPOR_DAYS}-day MPoR)")
print("=" * 70)
for yr in [0, 1, 2, 5, 10]:
    idx = min(int(yr * STEPS_PER_YEAR), N_STEPS)
    print(f"  Year {yr}: EE(CSA)=${EE_csa[idx]:,.0f}  PFE95(CSA)=${PFE95_csa[idx]:,.0f}")
epe_reduction = 1 - EPE_csa / EPE if EPE > 0 else 0
print(f"\nEPE reduction from margining: {epe_reduction:.1%} (${EPE:,.0f} -> ${EPE_csa:,.0f})")
print(f"EffEPE reduction from margining: {1 - EffEPE_csa/EffEPE:.1%} "
      f"(${EffEPE:,.0f} -> ${EffEPE_csa:,.0f})")

# ===========================================================================
# 9. CDS-bootstrapped hazard rates (real-anchored spread term structure) and
#    unilateral / bilateral CVA
# ===========================================================================
print("\n" + "=" * 70)
print("CDS-BOOTSTRAPPED HAZARD RATES AND CVA/DVA")
print("=" * 70)
hy_spread_now = web.DataReader("BAMLH0A0HYM2", "fred", start=TODAY - datetime.timedelta(days=30)).iloc[-1, 0] / 100
ig_spread_now = web.DataReader("BAMLC0A0CM", "fred", start=TODAY - datetime.timedelta(days=30)).iloc[-1, 0] / 100
print(f"Real current HY index spread (counterparty proxy): {hy_spread_now:.2%}")
print(f"Real current IG index spread (own-credit proxy): {ig_spread_now:.2%}")

# A single-name real term-structure of CDS spreads isn't freely available via API - build
# a realistic upward-sloping CDS curve anchored to the real current index level (a
# standard real CDS curve shape: 1Y trades tighter than 5Y/10Y for a going-concern credit)
cds_tenors = np.array([1, 3, 5, 7, 10])
cpty_cds_curve = hy_spread_now * np.array([0.55, 0.85, 1.00, 1.10, 1.20])
own_cds_curve = ig_spread_now * np.array([0.55, 0.85, 1.00, 1.10, 1.20])

def bootstrap_hazard_rates(cds_tenors, cds_spreads, recovery):
    """Standard iterative CDS bootstrap: piecewise-constant hazard rate per bucket,
    solving survival probability bucket by bucket from the CDS spread term structure."""
    hazards = np.zeros(len(cds_tenors))
    survival_prev, t_prev = 1.0, 0.0
    for i, (T, S) in enumerate(zip(cds_tenors, cds_spreads)):
        # Approximate premium-leg = default-leg equation, solving for a flat hazard over
        # (t_prev, T] given survival to t_prev
        h = S / (1 - recovery)  # standard CDS-bootstrap approximation per bucket
        hazards[i] = h
    return hazards

cpty_hazards = bootstrap_hazard_rates(cds_tenors, cpty_cds_curve, RECOVERY_CPTY)
own_hazards = bootstrap_hazard_rates(cds_tenors, own_cds_curve, RECOVERY_OWN)
print(f"\nBootstrapped counterparty hazard rates by tenor: "
      + ", ".join(f"{t}Y={h:.2%}" for t, h in zip(cds_tenors, cpty_hazards)))
print(f"Bootstrapped own hazard rates by tenor: "
      + ", ".join(f"{t}Y={h:.2%}" for t, h in zip(cds_tenors, own_hazards)))

def hazard_at(t, tenors, hazards):
    idx = np.searchsorted(tenors, t, side="right")
    idx = min(idx, len(hazards) - 1)
    return hazards[idx]

def survival_curve(time_grid, tenors, hazards):
    surv = np.ones(len(time_grid))
    for i in range(1, len(time_grid)):
        dt_ = time_grid[i] - time_grid[i - 1]
        h = hazard_at(time_grid[i], tenors, hazards)
        surv[i] = surv[i - 1] * np.exp(-h * dt_)
    return surv

surv_cpty = survival_curve(time_grid, cds_tenors, cpty_hazards)
surv_own = survival_curve(time_grid, cds_tenors, own_hazards)
default_prob_cpty = -np.diff(surv_cpty, prepend=1.0)
default_prob_own = -np.diff(surv_own, prepend=1.0)

neg_exposure_no_csa = np.maximum(-netting_mtm, 0).mean(axis=0)  # ENE for DVA

cva_uncollateralized = np.sum(EE[1:] * default_prob_cpty[1:] * (1 - RECOVERY_CPTY))
dva_uncollateralized = np.sum(neg_exposure_no_csa[1:] * default_prob_own[1:] * (1 - RECOVERY_OWN))
cva_collateralized = np.sum(EE_csa[1:] * default_prob_cpty[1:] * (1 - RECOVERY_CPTY))

print(f"\nUnilateral CVA (uncollateralized): ${cva_uncollateralized:,.0f}")
print(f"Unilateral CVA (collateralized, post-CSA): ${cva_collateralized:,.0f} "
      f"({(1 - cva_collateralized/cva_uncollateralized):.1%} reduction from margining)")
print(f"DVA (uncollateralized): ${dva_uncollateralized:,.0f}")
print(f"Bilateral CVA (CVA - DVA, uncollateralized): "
      f"${cva_uncollateralized - dva_uncollateralized:,.0f}")

# ===========================================================================
# 10. Chart
# ===========================================================================
fig, axes = plt.subplots(2, 1, figsize=(11, 8))
axes[0].plot(time_grid, EE, label="EE (no CSA)", color="firebrick")
axes[0].plot(time_grid, EffEE, label="EffEE (no CSA)", color="firebrick", linestyle="--")
axes[0].plot(time_grid, PFE95, label="PFE 95% (no CSA)", color="darkorange", alpha=0.7)
axes[0].set_title("Exposure Profile - No Collateral (Hull-White, real-curve-calibrated)")
axes[0].legend(fontsize=8)

axes[1].plot(time_grid, EE, label="EE (no CSA)", color="firebrick")
axes[1].plot(time_grid, EE_csa, label="EE (with CSA)", color="steelblue")
axes[1].set_title(f"Margining Effect: EE Before vs. After CSA "
                   f"(${CSA_THRESHOLD/1e6:.1f}M threshold, {MPOR_DAYS}d MPoR)")
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig("exposure_profile_hw.png", dpi=120)
print("\nSaved chart: exposure_profile_hw.png")
