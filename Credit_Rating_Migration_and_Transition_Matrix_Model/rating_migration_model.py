"""
Credit Rating Migration and Transition Matrix Model
======================================================
Uses S&P's real, publicly-published historical average one-year global corporate rating
transition matrix (the widely-cited table reproduced in S&P Global Ratings annual default
studies and standard credit-risk texts, e.g. de Servigny & Renault, "Measuring and
Managing Credit Risk") to project multi-year cumulative default probabilities via Markov
chain matrix exponentiation, simulate portfolio-level migration, and compute a CVA bump
from a simulated one-notch downgrade using a real market credit spread level (FRED).
"""
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import datetime

TODAY = datetime.date.today()

# ===========================================================================
# 1. Real S&P average one-year global corporate transition matrix (%)
#    Source: S&P Global Ratings Annual Global Corporate Default Study - the standard
#    published matrix reproduced across academic/regulatory credit risk literature.
#    Rows/cols: AAA, AA, A, BBB, BB, B, CCC/C, D
# ===========================================================================
ratings = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC/C", "D"]
transition_matrix = np.array([
    # AAA     AA      A      BBB     BB      B     CCC/C    D
    [90.63,  8.98,   0.30,  0.06,   0.02,   0.01,  0.00,   0.00],
    [0.59,   90.02,  8.85,  0.42,   0.08,   0.03,  0.01,   0.00],
    [0.04,   1.93,   91.19, 6.22,   0.44,   0.14,  0.02,   0.02],
    [0.02,   0.14,   3.93,  90.65,  4.24,   0.72,  0.13,   0.17],
    [0.02,   0.05,   0.24,  5.51,   84.16,  8.15,  0.60,   1.27],
    [0.00,   0.05,   0.19,  0.31,   5.55,   82.61, 4.63,   6.66],
    [0.00,   0.00,   0.24,  0.35,   1.07,   11.68, 51.19,  35.47],
    [0.00,   0.00,   0.00,  0.00,   0.00,   0.00,  0.00,   100.00],
]) / 100

df_matrix = pd.DataFrame(transition_matrix, index=ratings, columns=ratings)
print("Real S&P 1-year average global corporate transition matrix:")
print((df_matrix * 100).round(2).to_string())
print("\nSource: S&P Global Ratings Annual Global Corporate Default Study "
      "(standard published table)")

# ===========================================================================
# 2. Multi-year cumulative default probability via Markov chain exponentiation
# ===========================================================================
print("\n" + "=" * 70)
print("MULTI-YEAR CUMULATIVE DEFAULT PROBABILITY (Markov projection)")
print("=" * 70)
horizons = [1, 3, 5, 7, 10]
cum_pd = {}
for h in horizons:
    matrix_h = np.linalg.matrix_power(transition_matrix, h)
    cum_pd[h] = pd.Series(matrix_h[:, ratings.index("D")], index=ratings)

cum_pd_df = pd.DataFrame(cum_pd)
print((cum_pd_df * 100).round(3).to_string())
print("\n(Values = cumulative probability of default by year H, starting from each rating)")

# Sanity check against known real-world figures: S&P/Moody's-published 10-year cumulative
# default rates for BBB are typically in the 2-3% range, B in the 25-30% range
print(f"\nSanity check: 10Y cumulative PD for BBB = {cum_pd_df.loc['BBB', 10]:.2%} "
      f"(real-world published range: ~2-3%)")
print(f"Sanity check: 10Y cumulative PD for B = {cum_pd_df.loc['B', 10]:.2%} "
      f"(real-world published range: ~25-30%)")

# ===========================================================================
# 3. Simulate a sample portfolio's forward rating migration (Monte Carlo)
# ===========================================================================
np.random.seed(11)
N_BONDS = 100
N_YEARS = 5
starting_dist = {"AAA": 5, "AA": 10, "A": 25, "BBB": 35, "BB": 15, "B": 8, "CCC/C": 2}
portfolio = []
for rtg, count in starting_dist.items():
    portfolio += [rtg] * count
portfolio = np.array(portfolio)
assert len(portfolio) == N_BONDS

def simulate_migration(portfolio, matrix, ratings, years, seed_offset=0):
    rng = np.random.default_rng(seed_offset)
    current = portfolio.copy()
    history = [current.copy()]
    for y in range(years):
        new_ratings = []
        for rtg in current:
            if rtg == "D":
                new_ratings.append("D")
                continue
            idx = ratings.index(rtg)
            probs = matrix[idx]
            new_rtg = rng.choice(ratings, p=probs)
            new_ratings.append(new_rtg)
        current = np.array(new_ratings)
        history.append(current.copy())
    return history

history = simulate_migration(portfolio, transition_matrix, ratings, N_YEARS, seed_offset=11)

print("\n" + "=" * 70)
print(f"PORTFOLIO MIGRATION SIMULATION ({N_BONDS} bonds, {N_YEARS}-year horizon)")
print("=" * 70)
for y, snapshot in enumerate(history):
    counts = pd.Series(snapshot).value_counts().reindex(ratings, fill_value=0)
    print(f"Year {y}: " + " | ".join(f"{r}:{counts[r]}" for r in ratings))

start_rank = {r: i for i, r in enumerate(ratings)}
start_ranks = np.array([start_rank[r] for r in history[0]])
end_ranks = np.array([start_rank[r] for r in history[-1]])
upgrades = int((end_ranks < start_ranks).sum())
downgrades = int((end_ranks > start_ranks).sum())
defaults = int((history[-1] == "D").sum())
unchanged = N_BONDS - upgrades - downgrades
print(f"\nOver {N_YEARS} years: {upgrades} upgrades, {downgrades} downgrades "
      f"({defaults} of which defaulted), {unchanged} unchanged")

# Watch-list: bonds downgraded 2+ notches
watch_list = [(i, history[0][i], history[-1][i]) for i in range(N_BONDS)
              if start_rank.get(history[-1][i], 7) - start_rank[history[0][i]] >= 2]
print(f"Watch-list (downgraded 2+ notches over {N_YEARS}Y): {len(watch_list)} bonds")
for i, start, end in watch_list[:10]:
    print(f"  Bond #{i}: {start} -> {end}")

# ===========================================================================
# 4. CVA sensitivity to a 1-notch downgrade (real credit spread data, FRED)
# ===========================================================================
print("\n" + "=" * 70)
print("CVA SENSITIVITY TO RATING DOWNGRADE (real credit spread data)")
print("=" * 70)
ig_spread = web.DataReader("BAMLC0A0CM", "fred", start=TODAY - datetime.timedelta(days=30)).iloc[-1, 0] / 100
hy_spread = web.DataReader("BAMLH0A0HYM2", "fred", start=TODAY - datetime.timedelta(days=30)).iloc[-1, 0] / 100
print(f"Real IG credit spread (FRED BAMLC0A0CM): {ig_spread:.2%}")
print(f"Real HY credit spread (FRED BAMLH0A0HYM2): {hy_spread:.2%}")

# Approximate: CVA scales with hazard rate, hazard rate approximated from spread/(1-recovery)
recovery = 0.40
hazard_before = ig_spread / (1 - recovery)  # counterparty currently BBB (IG)
hazard_after = hy_spread / (1 - recovery)   # downgraded to BB (crosses into HY)

exposure_epe = 5_000_000  # from the CCR Exposure Engine project's average EPE, rounded
maturity = 5
cva_before = exposure_epe * hazard_before * maturity * (1 - recovery)
cva_after = exposure_epe * hazard_after * maturity * (1 - recovery)
print(f"\nCounterparty at BBB (IG), hazard rate {hazard_before:.2%}: CVA ~ ${cva_before:,.0f}")
print(f"After 1-notch downgrade to BB (HY), hazard rate {hazard_after:.2%}: CVA ~ ${cva_after:,.0f}")
print(f"CVA increase from 1-notch downgrade crossing IG/HY boundary: "
      f"{(cva_after/cva_before - 1):.1%} (${cva_after - cva_before:,.0f})")
