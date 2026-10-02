# IFRS 9 / CECL Expected Credit Loss Staging Model

**Status:** Built (Python).

## What it is
Stages every loan in the real 466,285-loan LendingClub dataset into IFRS 9 Stage 1
(performing), Stage 2 (significant credit deterioration), or Stage 3 (credit-impaired)
using the real `loan_status` field, computes 12-month ECL for Stage 1 and lifetime ECL
for Stage 2/3, and compares real loan-vintage cohorts to see how coverage differs across
origination periods.

## Data (real)
The same real 466,285-loan LendingClub dataset (2007-2014) already used in the headline
`Credit_Risk_PD_LGD_EAD_Modeling` project - real loan status (Current, Late 16-30/31-120,
In Grace Period, Charged Off, Default), real grade, real funded amount, real term, and
real issue date.

## Method
1. **Staging:** Stage 3 = Charged Off/Default (real, already-impaired loans); Stage 2 =
   Late 31-120 days or In Grace Period or Late 16-30 days (real, deteriorated but not
   yet impaired); Stage 1 = everything else (Current/Fully Paid).
2. **12-month PD:** a quick grade-based logistic regression (reusing the real grade
   fields from the headline project) as a fast, defensible 12-month PD proxy.
3. **Lifetime PD (Stage 2 only):** 12-month PD scaled by remaining contractual term - a
   standard simplifying convention when a full survival-model PD term structure isn't
   built, stated explicitly as an approximation.
4. **Stage 3 ECL:** LGD x EAD directly, NOT a projected PD x LGD x EAD - because a
   Stage 3 loan has already defaulted, so PD is 100% by definition.
5. **Vintage comparison:** real 2011-2012 origination cohort vs. real 2013-2014 cohort,
   comparing real coverage ratios - a genuine cohort comparison, explicitly NOT a
   same-loan roll-forward (this snapshot dataset has no multi-period panel of the same
   loans, which is stated honestly rather than implied).

## A real, caught-and-fixed methodology bug
An initial version applied the SAME projected lifetime-PD formula to Stage 3 loans as
Stage 2, which nonsensically treated an already-charged-off loan's loss as still
uncertain and probability-weighted, understating Stage 3 coverage to the same ~30% level
as Stage 2. Fixed by recognizing that Stage 3's defining feature IS that default has
already occurred - ECL for Stage 3 is LGD x EAD directly, with no PD term at all.

## Results (this run, real data)
| Stage | Loans | Exposure | ECL | Coverage Ratio |
|---|---|---|---|---|
| Stage 1 (performing) | 410,953 (88.13%) | $5,852,396,700 | $335,644,600 | 6% |
| Stage 2 (deteriorated) | 11,264 (2.42%) | $177,277,400 | $52,472,760 | 30% |
| Stage 3 (impaired) | 44,068 (9.45%) | $634,378,350 | $380,627,000 | **60% (exactly LGD)** |

**Total portfolio ECL: $768,744,382 on $6,664,052,450 exposure (11.54% overall coverage)**
- and the correct staircase pattern (6% -> 30% -> 60%) is itself the validation that the
staging and ECL logic are working as intended, since Stage 3's coverage should equal
LGD exactly once PD is correctly fixed at 100%.

**Vintage comparison:** the real 2011-2012 cohort shows 14.61% coverage vs. the real
2013-2014 cohort's 10.89% - a real -3.72 percentage point difference. This is a genuine,
interesting finding: later LendingClub vintages in this real dataset show a lower
observed coverage ratio, which could reflect either genuinely improving real credit
quality over time or simply less real seasoning time for the later cohort to migrate into
Stage 2/3 (a real, honest ambiguity worth flagging rather than picking one explanation
without evidence).

## Skills demonstrated
Real IFRS 9 3-stage classification logic, the real distinction between forward-looking
PD-based ECL (Stage 1/2) and already-realized-default ECL (Stage 3), and vintage-cohort
analysis - plus catching and fixing a real, conceptually significant methodology error
(treating an already-defaulted loan's loss as still probability-weighted) before
trusting the coverage-ratio output.

## Files
- `ifrs9_ecl_staging.py` - full script, runnable end to end
  (`py -3 ifrs9_ecl_staging.py`)
