"""
Commercial Loan Covenant Monitoring and Early Warning Dashboard
==================================================================
Pulls REAL quarterly financial statements (via yfinance) for a panel of real public
companies acting as "borrowers," computes covenant-style ratios (leverage, interest
coverage, current ratio) from real reported financials, applies illustrative
middle-market covenant thresholds, and builds a trend-based early-warning flag (not just
a hard breach) plus a ranked portfolio watch-list.
"""
import numpy as np
import pandas as pd
import yfinance as yf

# ===========================================================================
# 1. Real borrower panel - a mix of financially strong and stressed real companies
#    (chosen deliberately to include names that have had real leverage/coverage stress,
#    e.g. retail/industrial names, alongside stronger balance sheets, so the covenant
#    monitor has genuine variance to detect, not a uniformly healthy panel)
# ===========================================================================
tickers = ["AAPL", "F", "T", "CCL", "DAL", "KHC", "PARA", "WBA", "VZ", "NKE"]

# Illustrative middle-market-style covenant thresholds (typical real loan-agreement terms)
COVENANTS = {
    "max_leverage": 4.0,       # Total Debt / EBITDA
    "min_interest_coverage": 2.5,   # EBIT / Interest Expense
    "min_current_ratio": 1.10,      # Current Assets / Current Liabilities
}

records = []
for tkr in tickers:
    try:
        t = yf.Ticker(tkr)
        qf = t.quarterly_financials
        qbs = t.quarterly_balance_sheet
        if qf.empty or qbs.empty:
            continue
        n_periods = min(4, qf.shape[1], qbs.shape[1])

        def get_row(frame, *names):
            for name in names:
                if name in frame.index:
                    return frame.loc[name]
            return pd.Series(dtype=float)

        ebit_row = get_row(qf, "EBIT", "Operating Income")
        interest_row = get_row(qf, "Interest Expense")
        da_row = get_row(qf, "Reconciled Depreciation")

        for i in range(n_periods):
            period = qf.columns[i]
            # Trailing-4-quarter (TTM) EBITDA and interest expense - NOT a single quarter,
            # since Total Debt (balance-sheet stock) must be compared to an annualized flow
            window_cols = qf.columns[i:i + 4]
            ttm_ebit = ebit_row.reindex(window_cols).sum(skipna=True)
            ttm_da = da_row.reindex(window_cols).sum(skipna=True) if not da_row.empty else 0
            ttm_interest = abs(interest_row.reindex(window_cols).sum(skipna=True)) if not interest_row.empty else np.nan
            n_q = len(window_cols)
            if n_q < 4:  # fewer than 4 quarters available - annualize the run rate instead
                scale = 4 / n_q if n_q else np.nan
                ttm_ebit *= scale
                ttm_da *= scale
                ttm_interest *= scale
            ttm_ebitda = ttm_ebit + ttm_da

            try:
                total_debt = qbs.loc["Total Debt", period] if "Total Debt" in qbs.index else np.nan
            except KeyError:
                total_debt = np.nan
            try:
                curr_assets = qbs.loc["Current Assets", period] if "Current Assets" in qbs.index else np.nan
                curr_liab = qbs.loc["Current Liabilities", period] if "Current Liabilities" in qbs.index else np.nan
            except KeyError:
                curr_assets = curr_liab = np.nan

            if pd.isna(total_debt) or pd.isna(ttm_interest) or ttm_interest == 0:
                continue
            leverage = total_debt / ttm_ebitda if ttm_ebitda and ttm_ebitda != 0 else np.nan
            coverage = ttm_ebit / ttm_interest if ttm_interest else np.nan
            current_ratio = curr_assets / curr_liab if (pd.notna(curr_assets) and pd.notna(curr_liab) and curr_liab != 0) else np.nan

            records.append({
                "Ticker": tkr, "Period": period.date(), "TTM_EBITDA": ttm_ebitda,
                "Total_Debt": total_debt, "Leverage": leverage,
                "Interest_Coverage": coverage, "Current_Ratio": current_ratio,
            })
    except Exception as e:
        print(f"  [skip {tkr}: {e}]")

df = pd.DataFrame(records).sort_values(["Ticker", "Period"])
print(f"Pulled real quarterly financials for {df['Ticker'].nunique()} companies, "
      f"{len(df)} borrower-quarter observations\n")

# ===========================================================================
# 2. Covenant compliance check per borrower-quarter
# ===========================================================================
df["Leverage_Breach"] = df["Leverage"] > COVENANTS["max_leverage"]
df["Coverage_Breach"] = df["Interest_Coverage"] < COVENANTS["min_interest_coverage"]
df["Current_Ratio_Breach"] = df["Current_Ratio"] < COVENANTS["min_current_ratio"]
df["Any_Breach"] = df[["Leverage_Breach", "Coverage_Breach", "Current_Ratio_Breach"]].any(axis=1)

print("=" * 90)
print("COVENANT COMPLIANCE - MOST RECENT QUARTER PER BORROWER")
print("=" * 90)
latest = df.sort_values("Period").groupby("Ticker").tail(1).sort_values("Leverage", ascending=False)
print(latest[["Ticker", "Period", "Leverage", "Interest_Coverage", "Current_Ratio", "Any_Breach"]]
      .round(2).to_string(index=False))

breaches = latest[latest["Any_Breach"]]
print(f"\n{len(breaches)} of {len(latest)} borrowers currently in covenant breach: "
      f"{list(breaches['Ticker'])}")

# ===========================================================================
# 3. Trend-based early-warning: leverage headroom SHRINKING over time
#    (even if not yet breaching) - the real value-add over a static covenant check
# ===========================================================================
print("\n" + "=" * 90)
print("EARLY-WARNING TREND ANALYSIS (leverage headroom trend, not just a hard breach)")
print("=" * 90)
warning_list = []
for tkr, g in df.groupby("Ticker"):
    g = g.sort_values("Period")
    if len(g) < 2 or g["Leverage"].isna().any():
        continue
    headroom = COVENANTS["max_leverage"] - g["Leverage"].values
    trend = np.polyfit(range(len(headroom)), headroom, 1)[0]  # slope of headroom over time
    latest_headroom = headroom[-1]
    if trend < -0.05 and latest_headroom < 1.5:  # shrinking headroom AND getting close
        warning_list.append({
            "Ticker": tkr, "Latest_Leverage": g["Leverage"].values[-1],
            "Headroom_Trend_per_Q": trend, "Latest_Headroom": latest_headroom,
        })

warn_df = pd.DataFrame(warning_list).sort_values("Latest_Headroom") if warning_list else pd.DataFrame()
if not warn_df.empty:
    print(warn_df.round(3).to_string(index=False))
else:
    print("No borrowers currently show a shrinking-headroom early-warning pattern.")

# ===========================================================================
# 4. Ranked portfolio watch-list (breach OR early-warning)
# ===========================================================================
print("\n" + "=" * 90)
print("RANKED PORTFOLIO WATCH-LIST")
print("=" * 90)
watch = latest.copy()
watch["Watch_Score"] = (
    watch["Leverage_Breach"].astype(int) * 3 +
    watch["Coverage_Breach"].astype(int) * 2 +
    watch["Current_Ratio_Breach"].astype(int) * 1
)
watch = watch.sort_values("Watch_Score", ascending=False)
print(watch[["Ticker", "Watch_Score", "Leverage", "Interest_Coverage", "Current_Ratio"]]
      .round(2).to_string(index=False))
