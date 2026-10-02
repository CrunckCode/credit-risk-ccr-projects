# Counterparty Exposure Engine (Hull-White 1-Factor, Gregory-style)

**Status:** Built (Python).

## What it is
A Monte Carlo exposure engine for a 15-trade interest rate swap netting set: Hull-White
1-factor short-rate model with theta(t) fitted exactly to the real current curve, exact
analytic Hull-White zero-coupon bond pricing used to fully revalue every swap leg at every
time step (not a duration proxy), the full exposure profile (EE, EPE, EffEE, EffEPE, PFE
95%/99%), a real CSA collateral algorithm (threshold, MTA, margin period of risk), and
unilateral/bilateral CVA off hazard rates bootstrapped from a CDS-style spread curve.

## Data (real)
- **Hull-White mean reversion (a=0.5897) and volatility (sigma=0.6448%)** calibrated via
  AR(1) regression on 747 real daily SOFR observations (FRED `SOFR`, 2023-09-27 to
  2026-09-24).
- **Initial term structure** fitted to the real live Treasury/SOFR curve across 8 tenors
  (FRED `DGS3MO` through `DGS10`), 3-month 4.24% through 10Y 5.18%, via cubic-spline
  zero-yield interpolation.
- **r(0) = f(0,0)**, the model's own curve-implied instantaneous short rate (4.1976%),
  not a separately-observed market print - the correct, internally-consistent Hull-White
  convention.
- **CDS-style hazard-rate curve** anchored to real current credit-index spreads (FRED
  `BAMLH0A0HYM2` counterparty proxy 2.80%, `BAMLC0A0CM` own-credit proxy 0.79%), shaped
  into a realistic upward-sloping term structure and bootstrapped into piecewise hazard
  rates.

## Method
1. **theta(t)** computed via the exact closed-form Hull-White calibration formula
   (Brigo-Mercurio), so the model reproduces today's real forward curve exactly by
   construction - the defining feature that separates Hull-White from a plain Vasicek.
2. **Analytic bond pricing** P(t,T|r(t)) = A(t,T)e^(-B(t,T)r(t)) used to value every
   swap's fixed and floating legs exactly at every Monte Carlo node (2,000 paths, weekly
   steps, 10-year horizon), via the standard single-curve swap-valuation identity.
3. **Exposure profile**: EE (mean positive exposure), EffEE (running-max EE, Gregory's
   "effective EE"), EPE/EffEPE (time averages), PFE at 95%/99%.
4. **CSA collateral algorithm**: a $1M threshold, $250K minimum transfer amount, and a
   real 10-day margin period of risk (collateral lagged by the MPoR when computing the
   exposure a defaulting counterparty could actually realize).
5. **CVA/DVA**: hazard rates bootstrapped from the CDS-style curve, integrated against
   the EE profile for CVA and the negative-exposure profile for DVA.

## Three real bugs found and fixed during this build (the actual engineering story)
1. **r(0)/curve mismatch**: an early version set r(0) from the separate real SOFR print
   used for calibration rather than the model's own curve-implied f(0,0), producing a
   small but real nonzero netting-set inception value from a genuine SOFR-vs-Treasury
   basis. Fixed by setting r(0)=f(0,0) exactly, the correct Hull-White convention.
2. **MPoR-to-timestep conversion bug**: an initial days-to-steps formula was miscomputed
   and silently lagged the collateral by ~10 *months* instead of ~10 *days*, which
   (combined with then-monthly time steps) made the margined exposure profile look
   identical to the unmargined one for most of the horizon. Fixed by switching to weekly
   time steps (so a real 10-day MPoR maps to 1 real step) and correcting the day-count
   arithmetic.
3. **The real root cause of a large early exposure spike**: `theta(t)`'s calibration
   formula needs the *derivative* of the real forward curve, computed via finite
   difference - and a central difference evaluated at exactly t=0 secretly touches
   *negative* time, which the discount-factor floor turns into a near-zero forward rate
   and, through the derivative, an enormous spurious value of theta(0). That produced a
   huge one-off artificial rate shock on the very first Euler step, which showed up as a
   large deterministic (not random) exposure spike that at first glance looked like real
   path-dependent exposure buildup. Fixed with a one-sided (forward) finite difference for
   both f0(t) and its derivative whenever t is within one step of zero - the standard fix
   for finite-difference boundary artifacts.

## Results (this run, real-curve-calibrated, post-fix)
| Metric | No CSA | With CSA ($1M threshold, $250K MTA, 10d MPoR) |
|---|---|---|
| EPE | $188,872 | $164,870 (12.7% reduction) |
| EffEPE | $1,134,472 | $1,134,472 (0.0% reduction) |
| PFE99 (Year 1) | $1,858,258 | - |
| Unilateral CVA | $40,508 | $35,160 (13.2% reduction) |
| DVA | $26,677 | - |
| Bilateral CVA (CVA-DVA) | $13,831 | - |

**The real, non-obvious finding**: margining cuts average EPE by 12.7% but does almost
nothing for EffEPE (the peak-driven metric) - because the $1M threshold sits close to or
above this book's typical realized exposure (Year-1 EE of only ~$1.0-1.2M), so collateral
calls trigger rarely and the single worst-exposure moment across all 2,000 paths is barely
collateralized. **This is the same real lesson found independently in the earlier
Vasicek-based exposure project and the xVA pricer: a CSA threshold sized without reference
to the book's actual exposure distribution provides limited real protection** - here it
would need to sit well under $500K to meaningfully cap the peak metric, not just the
average one.

## Honesty note on scope
The 15-trade portfolio and CDS curve shape are constructed (real single-name CDS term
structures and real trustee-level swap books aren't freely available via API); the
short-rate calibration, initial curve, and credit-index spread anchors are all real and
live. Swap valuation assumes matched fixed/floating payment frequency for simplicity.

## Skills demonstrated
Full Hull-White 1-factor implementation (calibration, exact analytic bond pricing, proper
theta(t) fitting), Monte Carlo exposure simulation with genuine swap revaluation (not a
duration proxy), real CSA/MPoR margining mechanics, CDS-curve hazard-rate bootstrapping,
and - most importantly - a real debugging trail catching three distinct, explainable
implementation bugs (a curve-consistency error, a units-conversion error, and a
finite-difference boundary artifact) before trusting the model's output.

## Files
- `exposure_engine_hw.py` - full script, runnable end to end
  (`py -3 exposure_engine_hw.py`); recalibrates against live SOFR/Treasury/credit-spread
  data on every run
- `exposure_profile_hw.png` - exposure profile and margining-effect charts
