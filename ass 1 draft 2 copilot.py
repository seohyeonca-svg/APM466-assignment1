from datetime import date
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.interpolate import PchipInterpolator, interp1d

# =========================================
# GLOBAL SETTINGS / INPUTS
# =========================================

# Face value consistent with your price quotes
FACE = 100.0

# 10 pricing dates (Jan 5–9 and Jan 12–16, 2026)
PRICING_DATES = np.array(pd.to_datetime([
    "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09",
    "2026-01-12", "2026-01-13", "2026-01-14", "2026-01-15", "2026-01-16",
]).date)

# 10 maturities (in your order)
MATURITIES = np.array(pd.to_datetime([
    "2026-03-01", "2026-09-01",
    "2027-03-01", "2027-09-01",
    "2028-03-01", "2028-09-01",
    "2029-03-01", "2029-09-01",
    "2030-03-01", "2030-09-01",
]).date)

# Annual coupon rates as decimals (REPLACE THESE with your actual coupons if needed)
COUPON_RATES = np.array([0.0025, 0.0100, 0.0125, 0.0275, 0.0350, 0.0325, 0.0400, 0.0350, 0.0275, 0.0275
], dtype=float)

# Clean price lists (each must have length 10)
# >>> Replace these with YOUR real clean prices if needed.
CLEAN_PRICE_LISTS = [
    [99.70, 99.71, 99.71, 99.72, 99.73, 99.74, 99.74, 99.75, 99.76, 99.77],  # Bond 1
    [99.14, 99.14, 99.17, 99.16, 99.19, 99.18, 99.19, 99.20, 99.21, 99.22],  # Bond 2
    [98.60, 98.63, 98.66, 98.67, 98.67, 98.67, 98.68, 98.67, 98.73, 98.72],  # Bond 3
    [100.22, 100.30, 100.28, 100.31, 100.30, 100.32, 100.30, 100.31, 100.35, 100.37],  # Bond 4
    [101.73, 101.78, 101.78, 101.80, 101.79, 101.81, 101.78, 101.81, 101.84, 101.83],  # Bond 5
    [101.34, 101.41, 101.40, 101.43, 101.42, 101.45, 101.42, 101.43, 101.48, 101.47],  # Bond 6
    [103.63, 103.70, 103.71, 103.74, 103.73, 103.76, 103.72, 103.76, 103.79, 103.78],  # Bond 7
    [102.22, 102.33, 102.37, 102.34, 102.31, 102.35, 102.29, 102.33, 102.43, 102.42],  # Bond 8
    [99.49, 99.42, 99.56, 99.50, 99.58, 99.53, 99.50, 99.66, 99.66, 99.61],  # Bond 9
    [99.16, 99.08, 99.25, 99.17, 99.25, 99.21, 99.19, 99.36, 99.36, 99.31],  # Bond 10
]

# Interpolation method for the curve plots and interpolations
# "pchip" = smooth + no wiggles (recommended), or "linear"
INTERP_METHOD = "linear"

# =========================================
# 4 a) DIRTY PRICES + YTMs (cts comp)
# =========================================

def dirty(maturity_date: date, cpn_rate: float, clean_prices: np.ndarray,
          pricing_dates: np.ndarray, face: float) -> np.ndarray:
    """
    accrued interest = (n/365) * (face * cpn_rate)
    Dirty = Clean + accrued interest, where n = number of days since last coupon date.
    """
    dirty_prices = []

    for i, clean_price in enumerate(clean_prices):
        pricing_date = pricing_dates[i]

        # Find last coupon date (semiannual)
        last_cpn_date = maturity_date
        while last_cpn_date > pricing_date:
            last_cpn_date -= relativedelta(months=6)

        n = (pricing_date - last_cpn_date).days
        accrued_interest = (face * cpn_rate * n) / 365.0

        dirty_prices.append(clean_price + accrued_interest)

    return np.array(dirty_prices, dtype=float)


def cashflow_semiannual(maturity: date, settle: date):
    """All semiannual coupon dates after settlement up to maturity (inclusive)."""
    d = maturity
    while d > settle:
        d = d - relativedelta(months=6)
    d = d + relativedelta(months=6)

    dates = []
    while d <= maturity:
        dates.append(d)
        d = d + relativedelta(months=6)
    return dates


def setup_eqn(y: float, settle: date, maturity: date, cpn_rate: float, face: float) -> float:
    """
    Dirty price model using continuous compounding:
      P = sum CF_i * exp(-y * t_i)
    with t_i = days/365 (to match course convention).
    """
    cfd = cashflow_semiannual(maturity, settle)
    if len(cfd) == 0:
        return face

    coupon = face * cpn_rate / 2.0
    pv = 0.0

    for d in cfd:
        t = (d - settle).days / 365.0
        cf = coupon + (face if d == maturity else 0.0)
        pv += cf * np.exp(-y * t)

    return pv


def solve_ytm_continuous(dirty_price: float, settle: date, maturity: date, cpn_rate: float, face: float) -> float:
    """Solve for y in the continuous-compounding bond pricing equation via brentq."""
    def f(y):
        return setup_eqn(y, settle, maturity, cpn_rate, face) - dirty_price

    # wide bracket to allow negative yields
    y_lo, y_hi = -0.50, 2.00
    f_lo = f(y_lo)
    f_hi = f(y_hi)
    if f_lo * f_hi > 0:
        raise RuntimeError(f"Cannot bracket root for maturity={maturity}, settle={settle}. Check inputs.")
    return brentq(f, y_lo, y_hi, xtol=1e-12, maxiter=200)


# Convert your 10 lists into a (10,10) array: rows=days, cols=bonds
clean_prices = np.column_stack([np.array(lst, dtype=float) for lst in CLEAN_PRICE_LISTS])
if clean_prices.shape != (10, 10):
    raise ValueError(f"Expected 10 bonds each with 10 prices => shape (10,10). Got {clean_prices.shape}.")

# Compute dirty prices for each bond (10x10) and YTMs
dirty_prices = np.zeros((10, 10), dtype=float)
ytm = np.zeros((10, 10), dtype=float)
ttm = np.zeros((10, 10), dtype=float)  # time to maturity in years (days/365)

for j in range(10):
    dirty_prices[:, j] = dirty(
        maturity_date=MATURITIES[j],
        cpn_rate=float(COUPON_RATES[j]),
        clean_prices=clean_prices[:, j],
        pricing_dates=PRICING_DATES,
        face=FACE
    )

for i in range(10):
    settle = PRICING_DATES[i]
    for j in range(10):
        maturity = MATURITIES[j]
        ttm[i, j] = (maturity - settle).days / 365.0
        ytm[i, j] = solve_ytm_continuous(
            dirty_price=float(dirty_prices[i, j]),
            settle=settle,
            maturity=maturity,
            cpn_rate=float(COUPON_RATES[j]),
            face=FACE
        )

# Output table of YTMs
ytm_df = pd.DataFrame(
    ytm,
    index=pd.to_datetime(PRICING_DATES),
    columns=[f"Bond{j+1}_{MATURITIES[j]}" for j in range(10)]
)
print("\nContinuously-compounded YTMs (decimal):")
print(ytm_df.round(8))

# Plot 5-year yield curve for each day (superimposed)
T_grid = np.linspace(0.0, 5.0, 301)
plt.figure(figsize=(10, 6))
for i, settle in enumerate(pd.to_datetime(PRICING_DATES)):
    T = ttm[i, :]
    Y = ytm[i, :]
    # only maturities up to 5 years
    m = (T > 0) & (T <= 5.0)
    Tm, Ym = T[m], Y[m]
    order = np.argsort(Tm)
    Tm, Ym = Tm[order], Ym[order]
    if len(Tm) < 2:
        continue
    if INTERP_METHOD == "pchip":
        interp = PchipInterpolator(Tm, Ym, extrapolate=False)
        Yg = interp(T_grid)
    elif INTERP_METHOD == "linear":
        interp = interp1d(Tm, Ym, kind="linear", bounds_error=False, fill_value=np.nan)
        Yg = interp(T_grid)
    else:
        raise ValueError("INTERP_METHOD must be 'pchip' or 'linear'.")
    plt.plot(T_grid, Yg, linewidth=1.5, label=str(settle.date()))
plt.xlabel("Time to maturity (years, days/365)")
plt.ylabel("YTM (continuous compounding, decimal)")
plt.title("5-Year YTM Curves (10 days, superimposed)")
plt.grid(True, alpha=0.3)
plt.xlim(0, 5)
plt.legend(title="Pricing date", fontsize=8, ncols=2)
plt.tight_layout()
plt.show()

# =========================================
# 4 b) BOOTSTRAP SPOT CURVE (cts comp) + PLOT
# =========================================

def bootstrap_spot_curve_for_day(settle: date,
                                 maturities: np.ndarray,
                                 coupon_rates: np.ndarray,
                                 dirty_prices_day: np.ndarray,
                                 face: float):
    """
    Bootstrap discount factors DF(t) from shortest maturity upward, then spot rates:
      DF(t) = exp(-r(t)*t)  =>  r(t) = -ln(DF(t))/t  (continuous comp)
    Uses t = days/365, semiannual coupons.
    Returns sorted arrays (t_points, spot_points) for that given settle day.
    """
    df_map = {}  # key: time t, value: DF(t)
    order = np.argsort(maturities)
    maturities = maturities[order]
    coupon_rates = coupon_rates[order]
    dirty_prices_day = dirty_prices_day[order]

    for maturity, cpn, price in zip(maturities, coupon_rates, dirty_prices_day):
        cfd = cashflow_semiannual(maturity, settle)
        if len(cfd) == 0:     # already matured
            continue
        coupon = face * cpn / 2.0
        times = [(d - settle).days / 365.0 for d in cfd]
        cfs = [coupon + (face if d == maturity else 0.0) for d in cfd]

        # PV of known earlier cashflows using already-bootstrapped DFs
        pv_known = 0.0
        for t, cf in zip(times[:-1], cfs[:-1]):
            if t not in df_map:
                # try approx match due to float rounding
                matched = None
                for tk in df_map.keys():
                    if abs(tk - t) < 1e-10:
                        matched = tk
                        break
                if matched is None:
                    raise RuntimeError(
                        f"Missing DF for t={t:.12f} when bootstrapping maturity {maturity}. "
                        "Check maturity/coupon schedule consistency."
                    )
                t = matched
            pv_known += cf * df_map[t]

        T_last = times[-1]
        CF_last = cfs[-1]
        DF_last = (price - pv_known) / CF_last
        if DF_last <= 0:
            raise RuntimeError(
                f"Bootstrapping produced DF<=0 for maturity {maturity} on {settle}. "
                f"DF={DF_last}. Check prices/coupons/face."
            )
        df_map[T_last] = DF_last

    # Convert df_map to sorted spot points
    t_points = np.array(sorted(df_map.keys()), dtype=float)
    df_points = np.array([df_map[t] for t in t_points], dtype=float)
    spot_points = -np.log(df_points) / t_points  # r(t), continuous comp
    return t_points, spot_points


# ---- Compute spot curve (t_pts, r_pts) for each day
spot_rates_by_day = []
for i in range(10):
    settle = PRICING_DATES[i]
    t_pts, r_pts = bootstrap_spot_curve_for_day(
        settle=settle,
        maturities=MATURITIES,
        coupon_rates=COUPON_RATES,
        dirty_prices_day=dirty_prices[i, :],
        face=FACE
    )
    spot_rates_by_day.append((t_pts, r_pts))

# ---- Table of bootstrapped spot rates evaluated at each bond maturity time (per day)
spot_at_maturities = np.zeros((10, 10), dtype=float)
for i in range(10):
    Tm = ttm[i, :]  # maturity times for that day (already days/365)
    t_pts, r_pts = spot_rates_by_day[i]
    # Allow extrapolation to avoid NaNs at the right edge (e.g., ~>4.6y)
    if INTERP_METHOD == "pchip":
        spot_interp = PchipInterpolator(t_pts, r_pts, extrapolate=True)
    elif INTERP_METHOD == "linear":
        spot_interp = interp1d(t_pts, r_pts, kind="linear", bounds_error=False, fill_value="extrapolate")
    else:
        raise ValueError("INTERP_METHOD must be 'pchip' or 'linear'.")
    spot_at_maturities[i, :] = spot_interp(Tm)

spot_df = pd.DataFrame(
    spot_at_maturities,
    index=pd.to_datetime(PRICING_DATES),
    columns=[f"Spot@{MATURITIES[j]}" for j in range(10)]
)
print("\nBootstrapped continuously-compounded spot rates at each bond maturity (decimal):")
print(spot_df.round(8))

# ---- Plot 0–5y spot curves (superimposed)
T_grid = np.linspace(0.0, 5.0, 301)
plt.figure(figsize=(10, 6))
for i, settle_ts in enumerate(pd.to_datetime(PRICING_DATES)):
    t_pts, r_pts = spot_rates_by_day[i]
    m = (t_pts > 0) & (t_pts <= 5.0)
    t_plot = t_pts[m]
    r_plot = r_pts[m]
    if len(t_plot) < 2:
        continue
    if INTERP_METHOD == "pchip":
        interp = PchipInterpolator(t_plot, r_plot, extrapolate=False)
        Rg = interp(T_grid)
    elif INTERP_METHOD == "linear":
        interp = interp1d(t_plot, r_plot, kind="linear", bounds_error=False, fill_value=np.nan)
        Rg = interp(T_grid)
    else:
        raise ValueError("INTERP_METHOD must be 'pchip' or 'linear'.")
    plt.plot(T_grid, Rg, linewidth=1.5, label=str(settle_ts.date()))
plt.xlabel("Time (years, days/365)")
plt.ylabel("Spot rate r(t) (continuous compounding, decimal)")
plt.title("5-Year Spot Curves (Bootstrapped, 10 days, superimposed)")
plt.grid(True, alpha=0.3)
plt.xlim(0, 5)
plt.legend(title="Pricing date", fontsize=8, ncols=2)
plt.tight_layout()
plt.show()

# =========================================
# Helper: robust interpolator with controlled extrapolation
# =========================================

def make_spot_interpolator(t_pts, r_pts, method="linear", extrap="flat"):
    """
    Returns S(x) robust to out-of-bounds requests:
      - method: 'pchip' or 'linear'
      - extrap: 'flat' (edge value beyond range) or 'linear' (true extrapolation)
    Also removes NaNs and enforces strictly increasing unique x.
    """
    # keep only finite points
    m = np.isfinite(t_pts) & np.isfinite(r_pts)
    t = np.asarray(t_pts[m], float)
    r = np.asarray(r_pts[m], float)
    if t.size < 2:
        raise ValueError("Not enough points to build a spot interpolator.")

    # enforce strictly increasing unique x (round to tame float noise)
    t_rounded = np.round(t, 12)
    t_unique, idx = np.unique(t_rounded, return_index=True)
    t = t[idx]
    r = r[idx]

    # build core interpolator without extrapolation (returns NaN OOB)
    if method == "pchip":
        core = PchipInterpolator(t, r, extrapolate=False)
    elif method == "linear":
        core = interp1d(t, r, kind="linear", bounds_error=False, fill_value=np.nan, assume_sorted=False)
    else:
        raise ValueError("INTERP_METHOD must be 'pchip' or 'linear'.")

    t_min, t_max = t[0], t[-1]

    # optional true extrapolator
    if extrap == "linear":
        if method == "pchip":
            ext = PchipInterpolator(t, r, extrapolate=True)
        else:
            ext = interp1d(t, r, kind="linear", bounds_error=False, fill_value="extrapolate", assume_sorted=False)
    else:
        ext = None  # flat handled manually

    def S(x):
        x = np.asarray(x, float)
        y = core(x)
        # Fill NaNs caused by OOB using chosen extrapolation
        oob = ~np.isfinite(y)
        if np.any(oob):
            if extrap == "flat":
                y[oob & (x < t_min)] = r[0]
                y[oob & (x > t_max)] = r[-1]
            elif extrap == "linear":
                y[oob] = ext(x[oob])
            else:
                # leave NaN if extrap=None
                pass
        return y

    return S, (t_min, t_max)

# =========================================
# 4 c) 1-year forward curve (robust S(5))
# =========================================

# F_{t,t+n} = ( S_{t+n}*(t+n) - S_t*t ) / n
# Here n varies because we compute 1y-1y, 1y-2y, 1y-3y, 1y-4y -> end times 2,3,4,5 years.

FWD_N = 1.0
FORWARD_START = 1.0
FORWARD_ENDS = np.array([2.0, 3.0, 4.0, 5.0])  # curve points (end times)
fwd_cols = ["1y-1y", "1y-2y", "1y-3y", "1y-4y"]

# Choose extrapolation style for spots beyond last maturity:
#   'flat'   -> S(t>t_max) = S(t_max)  (conservative, avoids wild swings)
#   'linear' -> extend curve beyond last knot (trend-following)
EXTRAP_MODE = "flat"   # or "linear"

# Store forward rates per day: shape (10 days, 4 points)
fwd_curve = np.full((10, len(FORWARD_ENDS)), np.nan, dtype=float)

for i in range(10):
    t_pts, r_pts = spot_rates_by_day[i]
    # Robust spot interpolator
    S, (tmin, tmax) = make_spot_interpolator(
        t_pts, r_pts, method=INTERP_METHOD, extrap=EXTRAP_MODE
    )

    S1 = float(S(FORWARD_START))  # spot at 1y
    Stn = S(FORWARD_ENDS)         # vector: S(2), S(3), S(4), S(5)
    n = FORWARD_ENDS - FORWARD_START  # [1,2,3,4]
    fwd_curve[i, :] = (Stn * FORWARD_ENDS - S1 * FORWARD_START) / n

# ---- Table of forward curves
fwd_df = pd.DataFrame(
    fwd_curve,
    index=pd.to_datetime(PRICING_DATES),
    columns=fwd_cols
)
print("\n1-year forward curve points (continuous comp, decimal) [robust extrapolation]:")
print(fwd_df.round(8))

# ---- Plot forward curves (x-axis = end time or label)
plt.figure(figsize=(10, 6))
x = np.array([1, 2, 3, 4], dtype=float)  # positions for the 4 points
for i, settle_ts in enumerate(pd.to_datetime(PRICING_DATES)):
    plt.plot(x, fwd_curve[i, :], linewidth=1.5, marker="o", label=str(settle_ts.date()))
plt.xticks(x, fwd_cols)
plt.xlabel("Forward period (starting at 1 year)")
plt.ylabel("Forward rate (continuous compounding, decimal)")
plt.title(f"1-Year Forward Curves (from Spot Curve, 10 days, superimposed) — extrap='{EXTRAP_MODE}'")
plt.grid(True, alpha=0.3)
plt.legend(title="Pricing date", fontsize=8, ncols=2)
plt.tight_layout()
plt.show()

# =========================================
# 10) COVARIANCE MATRICES OF DAILY LOG-RETURNS
#     (A) yields at 1y..5y, (B) forward rates 1y-1y..1y-4y
# =========================================

# ----- A) Build 5 yield series r_i,j for i = 1..5 at maturities 1..5 years
YIELD_MATS = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)  # years
yields_1to5 = np.zeros((10, 5), dtype=float)  # rows=days, cols=1y..5y

for i in range(10):
    T = ttm[i, :]
    Y = ytm[i, :]

    m = (T > 0)
    Tm, Ym = T[m], Y[m]
    order = np.argsort(Tm)
    Tm, Ym = Tm[order], Ym[order]

    if INTERP_METHOD == "pchip":
        interpY = PchipInterpolator(Tm, Ym, extrapolate=True)
    elif INTERP_METHOD == "linear":
        interpY = interp1d(Tm, Ym, kind="linear", bounds_error=False, fill_value="extrapolate")
    else:
        raise ValueError("INTERP_METHOD must be 'pchip' or 'linear'.")
    yields_1to5[i, :] = interpY(YIELD_MATS)

yields_df = pd.DataFrame(
    yields_1to5,
    index=pd.to_datetime(PRICING_DATES),
    columns=["YTM_1y", "YTM_2y", "YTM_3y", "YTM_4y", "YTM_5y"]
)
print("\nInterpolated YTM series at 1y..5y (continuous comp, decimal):")
print(yields_df.round(8))

# ----- Daily log-returns for yields: X_{i,j} = log(r_{i,j+1}/r_{i,j})
yield_logret = np.log(yields_1to5[1:, :] / yields_1to5[:-1, :])  # shape (9,5)
yield_logret_df = pd.DataFrame(
    yield_logret,
    index=pd.to_datetime(PRICING_DATES[1:]),
    columns=["dlog_YTM_1y", "dlog_YTM_2y", "dlog_YTM_3y", "dlog_YTM_4y", "dlog_YTM_5y"]
)
print("\nDaily log-returns of yields (9 days):")
print(yield_logret_df.round(10))

# ----- Covariance matrix for yield log-returns (5x5)
cov_yield = np.cov(yield_logret.T, ddof=1)  # variables in columns => transpose
cov_yield_df = pd.DataFrame(
    cov_yield,
    index=["YTM_1y", "YTM_2y", "YTM_3y", "YTM_4y", "YTM_5y"],
    columns=["YTM_1y", "YTM_2y", "YTM_3y", "YTM_4y", "YTM_5y"]
)
print("\nCovariance matrix of daily log-returns of yields (5x5):")
print(cov_yield_df.round(12))

# ----- B) forward rates series already computed: fwd_curve (10x4)
fwd_df = pd.DataFrame(
    fwd_curve,
    index=pd.to_datetime(PRICING_DATES),
    columns=["F_1y1y", "F_1y2y", "F_1y3y", "F_1y4y"]
)
print("\nForward rate series (continuous comp, decimal):")
print(fwd_df.round(8))

# ----- Daily log-returns for forward rates (9x4)
fwd_logret = np.log(fwd_curve[1:, :] / fwd_curve[:-1, :])  # shape (9,4)
fwd_logret_df = pd.DataFrame(
    fwd_logret,
    index=pd.to_datetime(PRICING_DATES[1:]),
    columns=["dlog_F_1y1y", "dlog_F_1y2y", "dlog_F_1y3y", "dlog_F_1y4y"]
)
print("\nDaily log-returns of forward rates (9 days):")
print(fwd_logret_df.round(10))

# ----- Covariance matrix for forward log-returns (4x4)
cov_fwd = np.cov(fwd_logret.T, ddof=1)
cov_fwd_df = pd.DataFrame(
    cov_fwd,
    index=["F_1y1y", "F_1y2y", "F_1y3y", "F_1y4y"],
    columns=["F_1y1y", "F_1y2y", "F_1y3y", "F_1y4y"]
)
print("\nCovariance matrix of daily log-returns of forward rates (4x4):")
print(cov_fwd_df.round(12))

# =========================================
# 11) EIGENVALUES + EIGENVECTORS OF BOTH COVARIANCE MATRICES
# =========================================

def eigen_decomp_cov(cov_df: pd.DataFrame, name: str):
    """
    Eigen-decomposition for a symmetric covariance matrix.
    Uses np.linalg.eigh (best for symmetric matrices).
    Returns eigenvalues (descending) and eigenvectors (columns).
    """
    C = cov_df.values.astype(float)
    evals, evecs = np.linalg.eigh(C)   # ascending
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]  # columns are eigenvectors

    evals_s = pd.Series(evals, index=[f"PC{k+1}" for k in range(len(evals))], name=f"{name}_eigenvalue")
    evecs_df = pd.DataFrame(evecs, index=cov_df.index, columns=[f"PC{k+1}" for k in range(len(evals))])

    total = evals.sum()
    var_ratio = evals / total if total > 0 else np.nan
    var_ratio_s = pd.Series(var_ratio, index=evals_s.index, name=f"{name}_explained_variance_ratio")

    print(f"\n==== {name}: Eigenvalues ====")
    print(evals_s.round(12))
    print(f"\n==== {name}: Explained Variance Ratio ====")
    print(var_ratio_s.round(6))
    print(f"\n==== {name}: Eigenvectors (columns = PCs) ====")
    print(evecs_df.round(8))

    return evals_s, evecs_df, var_ratio_s

# --- Yield covariance eigendecomposition (5x5)
yield_evals, yield_evecs, yield_var_ratio = eigen_decomp_cov(cov_yield_df, "YIELD_COV")

# --- Forward covariance eigendecomposition (4x4)
fwd_evals, fwd_evecs, fwd_var_ratio = eigen_decomp_cov(cov_fwd_df, "FWD_COV")
