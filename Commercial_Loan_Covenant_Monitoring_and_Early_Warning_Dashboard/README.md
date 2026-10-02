# Commercial Loan Covenant Monitoring and Early Warning Dashboard

**Status:** Built (Python).

## What it is
A covenant-compliance and early-warning monitor built on real quarterly financial
statements of a panel of real public companies (used as borrower analogues), computing
leverage, interest coverage, and current ratio against illustrative middle-market
covenant thresholds, and - the actual point of the project - flagging borrowers whose
covenant *headroom is shrinking* over time, not just those already in breach.

## Data (real)
Real quarterly financial statements (`quarterly_financials`, `quarterly_balance_sheet` via
`yfinance`) for 7 real companies deliberately chosen to include genuine financial stress,
not a uniformly healthy panel: **F (Ford), VZ (Verizon), CCL (Carnival), KHC (Kraft
Heinz), DAL (Delta), T (AT&T), PARA (Paramount)**. Three other tickers pulled (AAPL, WBA,
NKE) had insufficient interest-expense line-item data in the available quarters and were
dropped automatically rather than filled with a placeholder.

## Method
1. Pull up to 4 quarters of real EBIT, interest expense, and depreciation/amortization,
   and real total debt / current assets / current liabilities from the balance sheet.
2. **Annualize correctly:** sum trailing-4-quarter EBIT+D&A and interest expense to a real
   TTM (trailing twelve months) basis before comparing to the balance-sheet debt stock -
   an early version of this script compared debt to a single quarter's EBITDA and produced
   nonsensical 15-700x leverage multiples; fixed by properly annualizing the flow measure.
3. Apply illustrative middle-market-style covenant thresholds (max leverage 4.0x, min
   interest coverage 2.5x, min current ratio 1.10x) to the most recent quarter per
   borrower.
4. **Early-warning trend check:** fit a linear trend to each borrower's covenant headroom
   (threshold minus actual leverage) over the available quarters; flag borrowers whose
   headroom is both shrinking and already thin, ahead of an actual breach.
5. Build a weighted watch-list score (leverage breach x3, coverage breach x2, current
   ratio breach x1) to rank borrowers by severity.

## Results (this run, 2026-09-26)
| Ticker | Leverage | Interest Coverage | Current Ratio | Watch Score |
|---|---|---|---|---|
| F (Ford) | 46.50x | -7.35x | 1.09 | 6 (worst) |
| PARA (Paramount) | -0.32x (negative EBITDA) | -10.97x | 0.10 | 3 |
| VZ, CCL, KHC, DAL, T | 3.0-3.9x | 3.6-9.2x | 0.33-1.06 | 1 each |

**Real finding on Ford:** its 46.5x leverage is a genuine artifact of captive auto
financing - Ford Motor Credit carries very large real debt relative to industrial-segment
EBIT, which is a widely-known real distortion in applying industrial covenant metrics to
auto OEMs with finance arms; a real credit analyst would strip out the captive-finance
segment before applying an industrial leverage covenant, exactly the kind of judgment
call this dashboard surfaces rather than hides.

**Early-warning finding:** VZ and KHC both show shrinking leverage headroom even though
neither has yet breached outright (headroom down to 0.08x and 0.45x respectively,
trending down $0.20x-$1.88x per quarter) - these are the two names a real portfolio
monitoring analyst would escalate before a covenant test actually fails, which is the
entire point of a trend-based early-warning system versus a static point-in-time check.

## Honesty note on scope
This panel is real large-cap public companies used as an illustrative borrower proxy
(middle-market/business-banking borrowers don't have public quarterly filings this
project can pull programmatically) - large corporates often run current ratios below 1.1
by deliberate working-capital policy, not distress, so the "current ratio breach" flag
here is noisier/less meaningful for this panel than it would be for an actual middle-market
borrower; the leverage and interest-coverage findings are the more reliable signal from
this data.

## Skills demonstrated
Real financial-statement ratio construction (with a caught-and-fixed annualization bug -
a genuine data-engineering judgment call), covenant-threshold compliance checking, and
trend-based (not just point-in-time) early-warning design - directly answers the
"reviews customer accounts and portfolios to identify... potential credit quality issues"
language common to Commercial/Business Banking Credit Analyst JDs.

## Files
- `covenant_monitoring.py` - full script, runnable end to end
  (`py -3 covenant_monitoring.py`); pulls fresh quarterly financials from Yahoo Finance on
  every run, so results update as new quarters are reported
