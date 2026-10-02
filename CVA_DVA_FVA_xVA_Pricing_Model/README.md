# CVA/DVA/FVA (xVA) Pricing Model

**Status:** Built (Python).

## What it is
The pricing layer on top of the Counterparty Credit Risk Exposure Engine: prices Credit
Valuation Adjustment (CVA), Debit Valuation Adjustment (DVA), and Funding Valuation
Adjustment (FVA) for the same netting set (2 IRS + 1 FX forward), using the same
real-SOFR-calibrated exposure profile, plus survival curves bootstrapped from real market
credit spreads for both the counterparty and the bank's own credit.

## Data (real)
- Reuses the real-SOFR-calibrated Vasicek short-rate model and real EURUSD volatility from
  the Exposure Engine project (kappa=0.590, theta=3.67%, sigma=0.64%).
- **Counterparty credit spread:** real ICE BofA High Yield index (FRED `BAMLH0A0HYM2`,
  2.80%) as a proxy for a sub-investment-grade counterparty's hazard rate.
- **Own (bank) credit spread:** real ICE BofA IG index (FRED `BAMLC0A0CM`, 0.79%) as a
  proxy for the bank's own funding/default curve.

## Method
1. Rebuild the exposure profile: positive exposure (EPE, bank's risk to counterparty
   default) and negative exposure (ENE, counterparty's risk to the bank's own default) at
   each future date across 5,000 Monte Carlo paths.
2. Bootstrap survival curves from the real credit spreads via the standard
   spread/(1-recovery) hazard-rate approximation (40% recovery assumption both sides).
3. **CVA** = sum over time of EPE(t) x marginal default probability(t) x loss-given-default.
4. **DVA** = symmetric calculation using ENE(t) and the bank's own survival curve - the
   standard "the bank benefits from its own default risk on this position" adjustment.
5. **FVA** = funding cost/benefit on the net funding exposure (EPE - ENE) at a 75bp
   unsecured funding spread, integrated over the exposure profile.
6. Recompute all three under a $2M CSA collateral threshold to show the mitigation effect.

## Results (this run)
| Metric | No CSA | With $2M CSA |
|---|---|---|
| CVA | $26,573 | $26,573 (0% reduction) |
| DVA | $9,173 | $9,173 (0% reduction) |
| FVA | -$1,040 | -$1,040 (0% reduction) |
| **Total xVA (CVA - DVA + FVA)** | **$16,359** | **$16,359** |

## Cross-project finding (consistent with the Exposure Engine project)
The CSA has **zero effect on any xVA component** for this netting set - because average
EPE never approaches the $2M threshold (this project's own EPE and the Exposure Engine's
PFE95 both stay under ~$1.4M at every horizon), the collateral floor never binds. This
directly confirms, from the pricing side, the same finding the Exposure Engine surfaced
from the exposure-profile side: **a CSA threshold sized without reference to the actual
exposure distribution provides no real risk or valuation benefit** - it would need to be
set meaningfully below ~$500K-$1M to actually reduce CVA/DVA/FVA for this book. Getting
the same answer from two independently-built models (exposure-based and pricing-based) is
itself a useful validation cross-check.

## Honesty note on scope
Uses index-level credit spreads (HY/IG benchmarks) as counterparty/own-credit proxies
rather than single-name CDS spreads (not freely available), and a simplified
unilateral-then-combined CVA/DVA calculation rather than a fully consistent bilateral xVA
framework with wrong-way-risk correlation. The mechanics (exposure-to-hazard-rate
integration, DVA symmetry, FVA funding-cost logic, CSA mitigation test) are real and
standard.

## Skills demonstrated
CVA/DVA/FVA pricing mechanics, survival-curve bootstrapping from real credit spreads,
funding-cost-based FVA calculation, and cross-validating a finding (CSA threshold
non-bindingness) across two independently-built models - exactly the xVA pricing
depth CCR JDs (Global Markets Risk - CPM, GRM Counterparty Credit Risk) ask for.

## Files
- `xva_pricing_model.py` - full script, runnable end to end
  (`py -3 xva_pricing_model.py`); recalibrates against live SOFR/EURUSD/credit-spread data
  on every run
