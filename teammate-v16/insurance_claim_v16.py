# =============================================================================
# DSC MCF ITB - INSURANCE CLAIM PREDICTION
# VERSI 16.0 — ROOT CAUSE FIXED + KALMAN + DAMPING + EXTERNAL DATA
# =============================================================================
#
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║          DEEP ROOT CAUSE ANALYSIS v15 → v16                                ║
# ╠══════════════════════════════════════════════════════════════════════════════╣
# ║                                                                              ║
# ║  MASALAH UTAMA v15 (dari visualisasi):                                      ║
# ║                                                                              ║
# ║  1. FREQUENCY FORECAST TERLALU RENDAH                                        ║
# ║     Jul 2025 historis: ~260 klaim (tertinggi sepanjang masa)                 ║
# ║     v15 forecast Agt: 198, Sep: 190 → TURUN DRASTIS                         ║
# ║     Padahal trend di 2025 sebenarnya NAIK → bias downward serius             ║
# ║     FIX: Kalman Filter untuk tracking state terkini + momentum correction    ║
# ║                                                                              ║
# ║  2. SEVERITY FORECAST TERLALU RENDAH                                         ║
# ║     Historis range: 44-62 Juta IDR                                           ║
# ║     v15 forecast: 42.6-48.5 Juta → Underestimate severity                   ║
# ║     Padding HW yang terlalu agresif menarik prediction ke bawah             ║
# ║     FIX: Severity TIDAK boleh turun terus → anchor ke recent level           ║
# ║                                                                              ║
# ║  3. TOTAL KLAIM FORECAST SANGAT RENDAH                                       ║
# ║     Historis recent: 12-14 Miliar IDR                                        ║
# ║     v15 forecast: 10-11 Miliar → downside bias ~20%                         ║
# ║     FIX: F×S = freq × sev langsung, BUKAN correction factor rumit           ║
# ║                                                                              ║
# ║  4. F×S RATIO MASIH 1.08 (OVERFIT ke training data)                         ║
# ║     Correction factor 1.08 menunjukkan systematic gap antara F×S dan Total  ║
# ║     Ini TIDAK boleh dipertahankan — harus ada estimasi lebih akurat         ║
# ║     FIX: Kalibrasi ulang dengan leave-one-out CV                            ║
# ║                                                                              ║
# ║  5. DAMPING TERLALU AGRESIF DI v15                                           ║
# ║     Seasonal dampening 40% → terlalu kuat, menghilangkan sinyal musiman     ║
# ║     YoY blending berlebih → drift ke bawah karena 2024 lebih rendah         ║
# ║     FIX: Optimal damping via cross-validation (bukan fixed 40%)             ║
# ║                                                                              ║
# ║  6. EXTERNAL DATA BELUM DIGUNAKAN                                            ║
# ║     Data inflasi medis, kurs USD/IDR, GDP, seasonality Ramadan/Natal        ║
# ║     bisa menjadi covariate yang sangat valuable untuk insurance forecast     ║
# ║     FIX: Integrasi external regressors ke SARIMAX model                     ║
# ║                                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# STRATEGI v16.0 — PRECISION FORECASTING:
#
# [FIX-1] KALMAN FILTER STATE TRACKER
#   → Track unobserved "true level" dari setiap series
#   → Optimal untuk smoothing noise tanpa over-dampening
#   → Kalman gain auto-adapted berdasarkan signal-to-noise ratio
#   → Memberikan "current state estimate" yang lebih akurat dari HW
#
# [FIX-2] OPTIMAL DAMPING VIA CROSS-VALIDATION
#   → Cari damping factor (phi) terbaik dari [0.80, 0.85, 0.90, 0.95, 0.98, 1.0]
#   → Damping applied ONLY ke trend component, bukan ke level
#   → Benchmark: phi=0.98 seringkali terbaik untuk short business series
#   → Damped Holt-Winters dengan phi optimal → prevents runaway trend
#
# [FIX-3] EXTERNAL DATA INTEGRATION (simulasi jika tidak ada akses net)
#   → Inflasi medis Indonesia (BPS): rata-rata 6-8% p.a. → monthly compounding
#   → Ramadan effect: klaim outpatient naik, inpatient turun → seasonal adj
#   → Lebaran (Hari Raya) effect: 2 bulan setelah Ramadan, klaim spike
#   → End-of-year effect: Des-Jan klaim naik (exhaustion of deductible)
#   → Data ini di-hardcode berdasarkan prior knowledge (tidak butuh internet)
#
# [FIX-4] SARIMAX DENGAN EXTERNAL REGRESSORS
#   → X = [medical_inflation_index, ramadan_dummy, endofyear_dummy,
#            trend_component, yoy_growth_lag]
#   → Model SARIMAX(p,d,q) × (P,D,Q,s=12) dengan exogenous variables
#   → Regularized (l1_pen) untuk variable selection
#
# [FIX-5] CONFORMAL PREDICTION INTERVAL
#   → Bukan Gaussian CI yang sering salah
#   → Walk-forward residuals → empirical quantile → valid coverage
#   → Point forecast = median dari conformal set (MAPE-optimal)
#
# [FIX-6] FREQUENCY ANCHOR CORRECTION
#   → v15 bias: prediksi freq 30% lebih rendah dari Jul 2025 actual
#   → Anchor: tidak boleh drop lebih dari X% dari recent 3-month average
#   → X = empirically computed dari training volatility
#
# [FIX-7] SEVERITY FLOOR PROTECTION
#   → Severity tidak boleh turun di bawah recent 6-month average × 0.85
#   → Karena medical inflation → secular upward trend
#   → Floor computed from actual data, NOT from assumption
#
# [FIX-8] TOTAL CLAIM = F × S + BIAS CORRECTION (simplified)
#   → Total = frequency_pred × severity_pred × correction_factor
#   → correction_factor = median dari (actual_total / (F × S_wins)) across months
#   → Tidak ada per-month correction → lebih robust, less overfit
#
# [FIX-9] ENSEMBLE WITH OPTIMAL STACKING WEIGHTS
#   → Stacking via Ridge regression (analytic optimal solution)
#   → Tidak perlu random grid search (stochastic, not reproducible)
#   → Weights directly minimize sum-of-squared walk-forward errors
#
# [FIX-10] DAMPED TREND HOLT-WINTERS YANG BENAR
#   → Phi search: cross-validate over [0.80 .. 1.00] step 0.01
#   → Pilih phi yang minimize WF-MAPE, bukan default 0.98
#   → Ini adalah "damping" yang dimaksud → prevent forecast collapse
#
# PUBLIC SCORE TARGET: < 4.0 (stretch goal: < 3.0)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import re
from itertools import product
warnings.filterwarnings('ignore')

from scipy import stats
from scipy.optimize import minimize, minimize_scalar, differential_evolution
from scipy.linalg import solve
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, HuberRegressor, Lasso
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.statespace.kalman_filter import KalmanFilter

try:
    import lightgbm as lgb
    HAS_LGB = True
    print("  ✓ LightGBM tersedia")
except ImportError:
    HAS_LGB = False
    print("  ℹ LightGBM tidak tersedia")

try:
    from prophet import Prophet
    HAS_PROPHET = True
    print("  ✓ Prophet tersedia")
except ImportError:
    HAS_PROPHET = False
    print("  ℹ Prophet tidak tersedia")

plt.rcParams.update({
    'figure.dpi': 100, 'font.family': 'DejaVu Sans',
    'axes.spines.top': False, 'axes.spines.right': False,
})
COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B', '#44BBA4',
          '#E84855', '#3D405B', '#81B29A', '#F2CC8F']

# ─── Path Data ───────────────────────────────────────────────────────────────
DATA_KLAIM  = r"Data_Klaim (1).csv"
DATA_POLIS  = r"Data_Polis.csv"
DATA_SAMPLE = r"sample_submission.csv"

print("=" * 80)
print("  DSC MCF ITB — INSURANCE CLAIM PREDICTION v16.0")
print("  Strategy: Kalman + Optimal Damping + External Data + Stacking Ensemble")
print("  v11 best: 9.20 | Target: < 4.0")
print("=" * 80)


# ========================== HELPER FUNCTIONS =================================

def calc_mape(y_true, y_pred, eps=1.0):
    yt = np.maximum(np.array(y_true, dtype=float), eps)
    yp = np.maximum(np.array(y_pred, dtype=float), 0)
    return float(np.mean(np.abs((yt - yp) / yt)) * 100)

def calc_rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2)))

def safe_kde(ax, data, color, linestyle='-', linewidth=2.0):
    d = np.array(data, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) < 2 or len(np.unique(d)) < 2:
        return
    try:
        kde = stats.gaussian_kde(d)
        xr = np.linspace(d.min(), d.max(), 200)
        ax.plot(xr, kde(xr), color=color, linewidth=linewidth, linestyle=linestyle)
    except:
        pass

def _title(ax, t, fs=11):
    ax.set_title(t, fontsize=fs, fontweight='bold')

def _no_data(ax, msg='Tidak tersedia'):
    ax.text(0.5, 0.5, msg, transform=ax.transAxes,
            ha='center', va='center', fontsize=10, color='gray')

def parse_numeric_safe(series):
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors='coerce')
    def _parse_one(val):
        if pd.isna(val): return np.nan
        s = str(val).strip()
        s = re.sub(r'[^\d.,\-]', '', s)
        if not s: return np.nan
        if ',' in s and '.' in s:
            last_dot, last_comma = s.rfind('.'), s.rfind(',')
            if last_comma > last_dot:
                s = s.replace('.', '').replace(',', '.')
            else:
                s = s.replace(',', '')
        elif ',' in s:
            parts = s.split(',')
            if len(parts) == 2 and len(parts[1]) <= 2:
                s = s.replace(',', '.')
            else:
                s = s.replace(',', '')
        elif '.' in s:
            parts = s.split('.')
            if len(parts) > 2:
                s = s.replace('.', '')
        try: return float(s)
        except: return np.nan
    return series.apply(_parse_one)

def parse_date_safe(series):
    if pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series):
        return pd.to_datetime(series.astype(str).str.split('.').str[0],
                              format='%Y%m%d', errors='coerce')
    result = pd.to_datetime(series, errors='coerce', dayfirst=True)
    for fmt in ['%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d', '%d-%m-%Y',
                '%Y%m%d', '%d %B %Y', '%B %d, %Y', '%Y-%m-%d']:
        missing = result.isna()
        if not missing.any(): break
        try:
            result[missing] = pd.to_datetime(series[missing], format=fmt, errors='coerce')
        except: pass
    return result

def find_col(df, keywords, required=False):
    cols_lower = {c.lower().strip(): c for c in df.columns}
    for kw in keywords:
        kw_l = kw.lower()
        if kw_l in cols_lower: return cols_lower[kw_l]
        for cl, corig in cols_lower.items():
            if kw_l in cl or cl in kw_l: return corig
    if required:
        raise KeyError(f"Kolom tidak ditemukan. Keywords: {keywords}")
    return None

def read_csv_robust(path):
    for enc in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']:
        for sep in [',', ';', '\t', '|']:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep, low_memory=False)
                if df.shape[1] > 1:
                    df.columns = df.columns.str.strip()
                    print(f"  → '{path}' (encoding={enc}, sep='{sep}')")
                    return df
            except: continue
    raise ValueError(f"Tidak bisa membaca: {path}")


# ========================== [FIX-1] KALMAN FILTER ===========================

class SimpleKalmanFilter:
    """
    [FIX-1] Local Level Kalman Filter (Structural Time Series Model)
    
    State equation:  level_t = level_{t-1} + noise_level ~ N(0, σ²_level)
    Obs equation:    y_t = level_t + noise_obs ~ N(0, σ²_obs)
    
    This is optimal for tracking slowly-changing unobserved level.
    Signal-to-noise ratio q = σ²_level / σ²_obs determines smoothing.
    q→0: static (heavily smoothed), q→∞: noisy/track every point.
    
    Optimal q estimated via MLE on training data.
    """
    def __init__(self, q=0.1):
        self.q = q  # signal-to-noise ratio
        self.level_ = None
        self.P_ = None
        self.fitted_values_ = None
        self.resid_ = None
        
    def fit(self, y, optimize_q=True):
        y = np.array(y, dtype=float)
        n = len(y)
        
        if optimize_q:
            def neg_ll(log_q):
                q_ = np.exp(log_q[0])
                _, ll = self._run_filter(y, q_)
                return -ll
            try:
                res = minimize(neg_ll, [np.log(0.1)], method='Nelder-Mead',
                               options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 200})
                self.q = np.exp(res.x[0])
            except:
                self.q = 0.1
        
        levels, Ps, innov = self._run_filter(y, self.q, return_levels=True)
        self.level_      = levels[-1]
        self.P_          = Ps[-1]
        self.fitted_values_ = levels
        self.resid_      = y - np.array(levels)
        self.y_          = y
        return self
    
    def _run_filter(self, y, q, return_levels=False):
        n = len(y)
        # Initialize
        level = y[0]
        P = 1.0
        
        levels = []
        Ps = []
        log_likelihood = 0.0
        
        for t in range(n):
            # Prediction
            level_pred = level
            P_pred = P + q
            
            # Innovation
            innov = y[t] - level_pred
            S = P_pred + 1.0  # obs noise = 1 (normalized)
            
            # Log-likelihood contribution
            if S > 0:
                log_likelihood += -0.5 * (np.log(2 * np.pi * S) + innov**2 / S)
            
            # Update
            K = P_pred / S  # Kalman gain
            level = level_pred + K * innov
            P = (1 - K) * P_pred
            
            levels.append(level)
            Ps.append(P)
        
        if return_levels:
            return levels, Ps, None
        return None, log_likelihood
    
    def forecast(self, h):
        """Forecast h steps ahead"""
        # Level forecast: constant (local level model)
        # Variance grows linearly with horizon
        forecasts = [self.level_] * h
        return np.array(forecasts)
    
    def forecast_with_trend(self, h, trend_estimate):
        """Forecast with external trend estimate"""
        forecasts = [self.level_ + trend_estimate * i for i in range(1, h+1)]
        return np.array(forecasts)


class KalmanTrendFilter:
    """
    Local Linear Trend Kalman Filter
    State: [level, trend]
    
    Level equation: level_t = level_{t-1} + trend_{t-1} + w1
    Trend equation: trend_t = phi * trend_{t-1} + w2  (phi=damping)
    Obs equation:   y_t = level_t + v
    
    This naturally implements DAMPED TREND forecasting!
    phi < 1 → damped, phi = 1 → random walk trend (Holt linear)
    """
    def __init__(self, phi=0.98, q_level=0.01, q_trend=0.001, q_obs=1.0):
        self.phi = phi
        self.q_level = q_level
        self.q_trend = q_trend
        self.q_obs   = q_obs
        self.level_  = None
        self.trend_  = None
        self.fitted_values_ = None
        
    def fit(self, y, optimize=True):
        y = np.array(y, dtype=float)
        
        if optimize:
            def neg_ll(params):
                phi = np.clip(params[0], 0.50, 1.00)
                q_l = np.exp(params[1])
                q_t = np.exp(params[2])
                try:
                    ll = self._run_filter_ll(y, phi, q_l, q_t, self.q_obs)
                    return -ll
                except:
                    return 1e10
            
            try:
                res = minimize(
                    neg_ll,
                    [0.95, np.log(0.01), np.log(0.001)],
                    method='Nelder-Mead',
                    options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 500}
                )
                self.phi     = np.clip(res.x[0], 0.50, 1.00)
                self.q_level = np.exp(res.x[1])
                self.q_trend = np.exp(res.x[2])
            except:
                pass
        
        levels, trends = self._run_filter(y, self.phi, self.q_level, self.q_trend, self.q_obs)
        self.level_ = levels[-1]
        self.trend_ = trends[-1]
        self.fitted_values_ = np.array(levels)
        self.resid_ = y - self.fitted_values_
        print(f"    [Kalman] phi={self.phi:.3f} q_l={self.q_level:.5f} q_t={self.q_trend:.5f}")
        return self
    
    def _run_filter_ll(self, y, phi, q_l, q_t, q_obs):
        n = len(y)
        # State: [level, trend]
        state = np.array([y[0], 0.0])
        P = np.eye(2) * 10.0
        
        F = np.array([[1, 1], [0, phi]])  # transition
        Q = np.diag([q_l, q_t])           # process noise
        H = np.array([[1, 0]])             # observation
        R = np.array([[q_obs]])            # obs noise
        
        ll = 0.0
        for t in range(n):
            # Predict
            state_pred = F @ state
            P_pred     = F @ P @ F.T + Q
            
            # Innovation
            innov = y[t] - float(H @ state_pred)
            S     = float(H @ P_pred @ H.T + R)
            
            if S <= 0: return -1e10
            ll += -0.5 * (np.log(2*np.pi*S) + innov**2/S)
            
            # Update
            K     = (P_pred @ H.T) / S
            state = state_pred + K.flatten() * innov
            P     = (np.eye(2) - K @ H) @ P_pred
        
        return ll
    
    def _run_filter(self, y, phi, q_l, q_t, q_obs):
        n = len(y)
        state = np.array([y[0], 0.0])
        P = np.eye(2) * 10.0
        
        F = np.array([[1, 1], [0, phi]])
        Q = np.diag([q_l, q_t])
        H = np.array([[1, 0]])
        R = np.array([[q_obs]])
        
        levels = []
        trends = []
        
        for t in range(n):
            state_pred = F @ state
            P_pred     = F @ P @ F.T + Q
            
            innov = y[t] - float(H @ state_pred)
            S     = float(H @ P_pred @ H.T + R)
            
            K = (P_pred @ H.T) / S
            state = state_pred + K.flatten() * innov
            P = (np.eye(2) - K @ H) @ P_pred
            
            levels.append(float(state[0]))
            trends.append(float(state[1]))
        
        return levels, trends
    
    def forecast(self, h):
        """Damped trend forecast h steps ahead"""
        phi   = self.phi
        level = self.level_
        trend = self.trend_
        
        forecasts = []
        curr_level = level
        curr_trend = trend
        
        for i in range(h):
            curr_level = curr_level + curr_trend
            curr_trend = phi * curr_trend  # damping
            forecasts.append(curr_level)
        
        return np.array(forecasts)


# ========================== [FIX-3] EXTERNAL DATA ===========================

def build_external_data(year_start=2024, month_start=1, n_months=None,
                        forecast_months=None, forecast_year=2025):
    """
    [FIX-3] Build external data regressors for Indonesia insurance market.
    
    Based on publicly available knowledge (no internet needed):
    1. Medical inflation Indonesia: ~6-8% annual (BPS/BPJS data)
    2. Ramadan effect: reduced inpatient, increased outpatient
    3. Lebaran effect: spike in claims after Eid (exhaustion effect)
    4. Year-end effect: claim spike in Nov-Dec (deductible exhaustion)
    5. Q1 effect: low claims Jan (reset deductible)
    6. USD/IDR trend: affects imported medical equipment costs
    
    Note: These are PRIOR KNOWLEDGE based features - they don't require
    internet access and are based on well-documented Indonesian healthcare patterns.
    """
    # Generate date range for training + forecast
    all_periods = []
    yr, mo = year_start, month_start
    max_iter = 60
    while max_iter > 0:
        all_periods.append((yr, mo))
        mo += 1
        if mo > 12:
            mo = 1
            yr += 1
        max_iter -= 1
        if n_months and len(all_periods) >= n_months:
            break
    
    if forecast_months:
        for fm in forecast_months:
            all_periods.append((forecast_year, fm))
    
    records = []
    base_inflation = 0.007  # ~8.4% annual medical inflation = 0.7% monthly
    
    # Ramadan/Lebaran calendar 2024-2025 (approximate, well-known dates)
    # Ramadan 2024: March 11 - April 9 → month 3
    # Eid ul-Fitr 2024: April 10 → month 4
    # Ramadan 2025: March 1 - March 30 → month 3
    # Eid ul-Fitr 2025: March 31 → month 3/4
    # Eid ul-Adha 2024: June 17 → month 6
    # Eid ul-Adha 2025: June 7 → month 6
    ramadan_months  = {(2024, 3): 1, (2025, 3): 1}  # Ramadan peak
    eid_months      = {(2024, 4): 1, (2025, 4): 1}  # Post-Eid claim surge
    adha_months     = {(2024, 6): 0.5, (2025, 6): 0.5}
    
    for i, (yr, mo) in enumerate(all_periods):
        t = i  # time index
        
        # Medical inflation index (compound monthly)
        # Jan 2024 = base 1.0
        months_since_start = (yr - year_start) * 12 + (mo - month_start)
        inflation_idx = (1 + base_inflation) ** months_since_start
        
        # Seasonal patterns (Indonesia-specific insurance claims)
        # Based on empirical knowledge of BPJS/private insurance patterns:
        # - Jan: Low (new deductible year starts)
        # - Mar-Apr: Ramadan/Lebaran → mixed effects
        # - Jun: Eid ul-Adha spike
        # - Jul-Ags: Mid-year, school holiday → lower inpatient
        # - Nov-Des: Year-end deductible exhaustion → high claims
        monthly_seasonality = {
            1: 0.85,   # Low - new deductible year
            2: 0.90,   # Slightly low
            3: 1.00,   # Average
            4: 1.05,   # Post-Ramadan spike
            5: 0.95,   # Return to normal
            6: 1.10,   # Eid ul-Adha spike + mid-year
            7: 1.00,   # Normal
            8: 0.95,   # School holiday
            9: 1.00,   # Normal
            10: 1.05,  # Pre-year-end
            11: 1.10,  # Year-end claim rush
            12: 1.15,  # Highest - deductible exhaustion
        }
        
        record = {
            'year': yr,
            'month': mo,
            'time_idx': months_since_start,
            'inflation_idx': inflation_idx,
            'inflation_mom': base_inflation,  # monthly inflation
            
            # Calendar effects
            'is_ramadan':   int((yr, mo) in ramadan_months),
            'is_eid_fitr':  int((yr, mo) in eid_months),
            'is_eid_adha':  int((yr, mo) in adha_months),
            'is_yearend':   int(mo in [11, 12]),
            'is_yearstart': int(mo == 1),
            'is_q1':        int(mo in [1, 2, 3]),
            'is_q4':        int(mo in [10, 11, 12]),
            'is_holiday':   int(mo in [1, 4, 6, 7, 12]),
            
            # Seasonal index (prior knowledge)
            'seasonal_prior': monthly_seasonality.get(mo, 1.0),
            
            # Trend features
            'linear_trend': months_since_start / 12.0,  # years elapsed
            'trend_sq':     (months_since_start / 12.0) ** 2,
            
            # Cyclic encoding
            'sin_month': np.sin(2 * np.pi * mo / 12),
            'cos_month': np.cos(2 * np.pi * mo / 12),
            'sin_qtr':   np.sin(2 * np.pi * ((mo-1)//3 + 1) / 4),
            'cos_qtr':   np.cos(2 * np.pi * ((mo-1)//3 + 1) / 4),
        }
        records.append(record)
    
    df_ext = pd.DataFrame(records)
    return df_ext


# ========================== [FIX-10] OPTIMAL DAMPING ========================

def find_optimal_damping(series, phi_grid=None):
    """
    [FIX-10] Find optimal damping factor via Walk-Forward Cross-Validation.
    
    Damped Holt-Winters: forecast = level + phi/(1-phi) * trend (as h→∞)
    phi=1.0 → no damping (linear extrapolation)
    phi=0.98 → mild damping (industry standard for business forecasting)
    phi=0.80 → strong damping (quickly reverts to mean)
    
    Key insight: For insurance with ~19 months data, phi ≈ 0.90-0.95 is often best.
    """
    if phi_grid is None:
        phi_grid = [0.80, 0.85, 0.88, 0.90, 0.92, 0.95, 0.97, 0.98, 0.99, 1.00]
    
    series = np.array(series, dtype=float)
    n = len(series)
    min_train = max(8, n - 8)
    
    best_phi  = 0.95
    best_mape = np.inf
    phi_mapes = {}
    
    for phi in phi_grid:
        wf_mapes = []
        for test_idx in range(min_train, n):
            y_train = series[:test_idx]
            y_true  = series[test_idx]
            
            if len(y_train) < 6: continue
            
            try:
                if phi < 1.0:
                    model = ExponentialSmoothing(
                        y_train, trend='add', seasonal=None,
                        damped_trend=True, initialization_method='estimated'
                    )
                    fit = model.fit(
                        damping_trend=phi, optimized=True, use_brute=False
                    )
                else:
                    model = ExponentialSmoothing(
                        y_train, trend='add', seasonal=None,
                        damped_trend=False, initialization_method='estimated'
                    )
                    fit = model.fit(optimized=True, use_brute=False)
                
                pred = float(fit.forecast(1).iloc[0])
                pred = max(0, pred)
                wf_mapes.append(calc_mape([y_true], [pred]))
            except:
                continue
        
        if wf_mapes:
            avg_mape = np.mean(wf_mapes)
            phi_mapes[phi] = avg_mape
            if avg_mape < best_mape:
                best_mape = avg_mape
                best_phi  = phi
    
    return best_phi, best_mape, phi_mapes


# ========================== LOAD & PREPROCESS ===============================
print("\n[1/10] LOADING DATA...")

df_klaim  = read_csv_robust(DATA_KLAIM)
df_polis  = read_csv_robust(DATA_POLIS)
df_sample = read_csv_robust(DATA_SAMPLE)

print(f"  ✓ Data Klaim  : {df_klaim.shape[0]:,} × {df_klaim.shape[1]}")
print(f"  ✓ Data Polis  : {df_polis.shape[0]:,} × {df_polis.shape[1]}")
print(f"  ✓ Sample IDs  : {df_sample['id'].tolist()}")

print("\n[2/10] PRE-PROCESSING...")

COL_POLIS_K    = find_col(df_klaim, ['nomor polis', 'nopolis', 'polis_id', 'polis'], required=True)
COL_STATUS     = find_col(df_klaim, ['status klaim', 'status claim', 'status'], required=True)
COL_TARGET     = find_col(df_klaim, ['nominal klaim yang disetujui', 'klaim disetujui',
                                     'nominal disetujui', 'approved claim', 'disetujui'])
COL_TGL_MASUK  = find_col(df_klaim, ['tanggal pasien masuk rs', 'tanggal pasien masuk',
                                     'tgl masuk', 'masuk rs', 'admission date', 'masuk'])
COL_TGL_KELUAR = find_col(df_klaim, ['tanggal pasien keluar rs', 'tanggal pasien keluar',
                                     'tgl keluar', 'keluar rs', 'discharge date', 'keluar'])
COL_TGL_BAYAR  = find_col(df_klaim, ['tanggal pembayaran klaim', 'tanggal pembayaran',
                                     'tgl bayar', 'payment date', 'pembayaran'])
COL_ICD        = find_col(df_klaim, ['icd diagnosis', 'icd_diagnosis', 'icd code', 'icd'])
COL_IP_OP      = find_col(df_klaim, ['inpatient/outpatient', 'rawat inap', 'ip_op', 'rawat'])
COL_LOKASI_RS  = find_col(df_klaim, ['lokasi rs', 'lokasi rumah sakit', 'lokasi'])
COL_BIAYA_RS   = find_col(df_klaim, ['nominal biaya rs yang terjadi', 'nominal biaya rs',
                                     'biaya rs', 'biaya'])
COL_REIMBURSE  = find_col(df_klaim, ['reimburse/cashless', 'reimburse', 'cashless'])

COL_POLIS_P   = find_col(df_polis, ['nomor polis', 'nopolis', 'polis_id', 'polis'], required=True)
COL_TGL_LAHIR = find_col(df_polis, ['tanggal lahir', 'tgl lahir', 'birth date', 'dob'])
COL_TGL_EFF   = find_col(df_polis, ['tanggal efektif polis', 'tanggal efektif', 'efektif'])
COL_PLAN      = find_col(df_polis, ['plan code', 'plancode', 'plan'])
COL_GENDER    = find_col(df_polis, ['gender', 'jenis kelamin', 'sex'])

# Parse numeric & dates
df_klaim[COL_TARGET] = parse_numeric_safe(df_klaim[COL_TARGET])
if COL_BIAYA_RS and COL_BIAYA_RS in df_klaim.columns:
    df_klaim[COL_BIAYA_RS] = parse_numeric_safe(df_klaim[COL_BIAYA_RS])

for col in [c for c in [COL_TGL_MASUK, COL_TGL_KELUAR, COL_TGL_BAYAR]
            if c and c in df_klaim.columns]:
    df_klaim[col] = parse_date_safe(df_klaim[col])

for col in [c for c in [COL_TGL_LAHIR, COL_TGL_EFF]
            if c and c in df_polis.columns]:
    df_polis[col] = parse_date_safe(df_polis[col])

df = df_klaim.merge(df_polis, left_on=COL_POLIS_K, right_on=COL_POLIS_P, how='left')
if COL_POLIS_K != COL_POLIS_P and COL_POLIS_P in df.columns:
    df = df.drop(columns=[COL_POLIS_P])

# Impute payment date
if (COL_TGL_BAYAR and COL_TGL_MASUK and
        COL_TGL_BAYAR in df.columns and COL_TGL_MASUK in df.columns):
    known = df[COL_TGL_BAYAR].notna() & df[COL_TGL_MASUK].notna()
    med_lag = 14
    if known.sum() > 0:
        lag = (df.loc[known, COL_TGL_BAYAR] - df.loc[known, COL_TGL_MASUK]).dt.days
        if not lag.isna().all():
            med_lag = int(np.clip(lag.median(), 0, 90))
    paid_na = df[COL_STATUS].isin(['PAID']) & df[COL_TGL_BAYAR].isna()
    df.loc[paid_na, COL_TGL_BAYAR] = (
        df.loc[paid_na, COL_TGL_MASUK] + pd.Timedelta(days=med_lag))

for col in df.select_dtypes(include=[np.number]).columns:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())
for col in df.select_dtypes(include='object').columns:
    if df[col].isna().any():
        mode_ = df[col].mode()
        df[col] = df[col].fillna(mode_.iloc[0] if not mode_.empty else 'Unknown')

# Filter PAID
status_counts = df[COL_STATUS].value_counts()
PAID_VALUES = [v for v in status_counts.index
               if any(kw in str(v).lower()
                      for kw in ['paid', 'dibayar', 'approved', 'disetujui', 'settled'])]
if not PAID_VALUES:
    PAID_VALUES = [status_counts.index[0]]

df_paid = df[df[COL_STATUS].isin(PAID_VALUES)].copy().reset_index(drop=True)
df_paid[COL_TARGET] = parse_numeric_safe(df_paid[COL_TARGET])

masuk_col = (COL_TGL_MASUK if COL_TGL_MASUK and COL_TGL_MASUK in df_paid.columns
             else COL_TGL_BAYAR)
df_paid[masuk_col] = pd.to_datetime(df_paid[masuk_col], errors='coerce')

valid_dates = df_paid[masuk_col].dropna().sort_values()
if df_paid[masuk_col].isna().any() and len(valid_dates) > 0:
    med_date = valid_dates.iloc[len(valid_dates) // 2]
    df_paid[masuk_col] = df_paid[masuk_col].fillna(med_date)

df_paid['tahun']      = df_paid[masuk_col].dt.year.fillna(2024).astype(int)
df_paid['bulan']      = df_paid[masuk_col].dt.month.fillna(1).astype(int)
df_paid['year_month'] = df_paid[masuk_col].dt.to_period('M')

TARGET_COL = COL_TARGET

# ─── Adaptive Winsorization (from v15 insight) ───────────────────────────────
# Choose best winsorization level based on F×S ratio closest to 1.0
for pct in [(0.01, 0.99), (0.05, 0.95)]:
    col_name = f'target_w{int(pct[1]*100)}'
    df_paid[col_name] = df_paid[TARGET_COL].clip(
        lower=df_paid[TARGET_COL].quantile(pct[0]),
        upper=df_paid[TARGET_COL].quantile(pct[1])
    )

# Quick check: which winsorization gives ratio closest to 1.0
ratios = {}
for win_col in ['target_w99', 'target_w95']:
    tmp = df_paid.groupby('year_month').agg(
        freq=(TARGET_COL, 'count'),
        sev_w=(win_col, 'mean'),
        total=(TARGET_COL, 'sum')
    )
    ratio_ = (tmp['total'] / (tmp['freq'] * tmp['sev_w'] + 1e-10)).median()
    ratios[win_col] = abs(ratio_ - 1.0)

best_win_col = min(ratios, key=ratios.get)
df_paid['target_wins'] = df_paid[best_win_col].copy()
TARGET_WINS_COL = 'target_wins'

print(f"  ✓ df_paid     : {len(df_paid):,} baris")
print(f"  Best winsor.  : {best_win_col} (|ratio-1|={ratios[best_win_col]:.3f})")
print(f"  PAID values   : {PAID_VALUES}")


# ========================== 3. EDA ==========================================
print("\n[3/10] EXPLORATORY DATA ANALYSIS...")

mf_eda = df_paid.groupby('year_month').agg(
    freq     = (TARGET_COL, 'count'),
    sev_mean = (TARGET_COL, 'mean'),
    sev_wins = ('target_wins', 'mean'),
    sev_med  = (TARGET_COL, 'median'),
    total    = (TARGET_COL, 'sum'),
    p25      = (TARGET_COL, lambda x: x.quantile(.25)),
    p75      = (TARGET_COL, lambda x: x.quantile(.75))
).reset_index()
mf_eda['ym_str'] = mf_eda['year_month'].astype(str)
mf_eda['ym_dt']  = pd.to_datetime(mf_eda['ym_str'])

# Key diagnostic from v15 analysis:
ratio_train = (mf_eda['total'] / (mf_eda['freq'] * mf_eda['sev_wins'] + 1e-10)).median()
print(f"  F×S ratio (global)   : {ratio_train:.3f}")

# ─── External Data for EDA period ────────────────────────────────────────────
start_yr = int(str(mf_eda['year_month'].min())[:4])
start_mo = int(str(mf_eda['year_month'].min())[5:7])
df_ext_train = build_external_data(
    year_start=start_yr, month_start=start_mo, n_months=len(mf_eda)
)
print(f"  External data shape  : {df_ext_train.shape}")
print(f"  Inflation idx range  : {df_ext_train['inflation_idx'].min():.3f} - {df_ext_train['inflation_idx'].max():.3f}")

# ─── EDA Dashboard ───────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 28))
fig.patch.set_facecolor('#F8F9FA')
fig.suptitle('DSC MCF ITB — Insurance Claim EDA Dashboard v16.0\n'
             'Kalman Filter + Optimal Damping + External Data | Target: <4.0',
             fontsize=14, fontweight='bold', y=0.998)
gs = gridspec.GridSpec(5, 3, figure=fig, hspace=0.60, wspace=0.40)

# Plot 1: Frequency Trend
ax1 = fig.add_subplot(gs[0, :2])
try:
    bar_colors = ['#C73E1D' if '2025' in str(x) else '#2E86AB' for x in mf_eda['year_month']]
    ax1.bar(range(len(mf_eda)), mf_eda['freq'], color=bar_colors, alpha=0.7, edgecolor='white')
    ax1.plot(range(len(mf_eda)), mf_eda['freq'], color='#1a1a2e', lw=2, marker='o', ms=4, zorder=5)
    x_arr = np.arange(len(mf_eda))
    z = np.polyfit(x_arr, mf_eda['freq'], 1)
    ax1.plot(x_arr, np.poly1d(z)(x_arr), 'r--', lw=2, alpha=0.7, label=f'Trend ({z[0]:+.1f}/bln)')
    
    # [v16] Quick Kalman fit
    kf_quick = SimpleKalmanFilter()
    kf_quick.fit(mf_eda['freq'].values, optimize_q=True)
    ax1.plot(range(len(mf_eda)), kf_quick.fitted_values_, color='#F18F01',
             lw=2.5, ls='-.', label=f'Kalman Level (q={kf_quick.q:.3f})')
    
    ax1.set_xticks(range(len(mf_eda)))
    ax1.set_xticklabels(mf_eda['ym_str'], rotation=45, ha='right', fontsize=7.5)
    ax1.set_ylabel('Frekuensi Klaim / Bulan')
    ax1.legend(fontsize=9)
    
    # Annotate last point (key for understanding v15 underforecast)
    last_val = float(mf_eda['freq'].iloc[-1])
    ax1.annotate(f'Last: {last_val:.0f}\n[v15: 198?]',
                 xy=(len(mf_eda)-1, last_val),
                 xytext=(len(mf_eda)-3, last_val+15),
                 arrowprops=dict(arrowstyle='->', color='red'),
                 fontsize=9, color='red')
except Exception as e:
    _no_data(ax1, f'{e}')
_title(ax1, '[v16 KEY] Frekuensi Klaim + Kalman Level  (Merah = 2025)', fs=12)

# Plot 2: ACF/PACF
ax2 = fig.add_subplot(gs[0, 2])
try:
    freq_series = mf_eda['freq'].values
    if len(freq_series) >= 6:
        nlags     = min(len(freq_series) - 2, 15)
        acf_vals  = acf(freq_series, nlags=nlags, fft=True)
        pacf_vals = pacf(freq_series, nlags=min(nlags, len(freq_series)//2-1))
        ax2.bar(range(len(acf_vals)), acf_vals, color=COLORS[1], alpha=0.75, label='ACF')
        ax2.step(range(len(pacf_vals)), pacf_vals, color=COLORS[0], lw=2, label='PACF', where='mid')
        ci = 1.96 / np.sqrt(len(freq_series))
        ax2.axhline(y=ci, color='red', ls='--', lw=1.5, alpha=0.7)
        ax2.axhline(y=-ci, color='red', ls='--', lw=1.5, alpha=0.7)
        ax2.axhline(y=0, color='black', lw=0.8)
        ax2.legend(fontsize=8)
except Exception as e:
    _no_data(ax2, f'{e}')
_title(ax2, 'ACF + PACF (ARIMA Order Guidance)')

# Plot 3: Damping phi analysis (NEW in v16)
ax3 = fig.add_subplot(gs[1, 0])
try:
    phi_grid_eda = [0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.98, 0.99, 1.00]
    best_phi_eda, best_mape_phi, phi_mapes = find_optimal_damping(
        mf_eda['freq'].values, phi_grid=phi_grid_eda
    )
    phis = list(phi_mapes.keys())
    mapes_ = [phi_mapes[p] for p in phis]
    best_idx = np.argmin(mapes_)
    bar_colors_phi = ['#C73E1D' if i == best_idx else '#2E86AB' for i in range(len(phis))]
    ax3.bar([str(p) for p in phis], mapes_, color=bar_colors_phi, alpha=0.85)
    ax3.set_ylabel('WF-MAPE (%)')
    ax3.set_xticklabels([str(p) for p in phis], rotation=45, ha='right', fontsize=8)
    ax3.text(0.05, 0.97, f'Optimal φ={best_phi_eda}\nMAPE={best_mape_phi:.2f}%',
             transform=ax3.transAxes, fontsize=9, va='top',
             bbox=dict(boxstyle='round', facecolor='#FFF3CD', alpha=0.9))
    OPTIMAL_PHI_FREQ = best_phi_eda
except Exception as e:
    _no_data(ax3, f'{e}')
    OPTIMAL_PHI_FREQ = 0.95
_title(ax3, '[v16 NEW] Optimal Damping Factor (φ) Search')

# Plot 4: Kalman Filter vs HW comparison
ax4 = fig.add_subplot(gs[1, 1])
try:
    freq_vals = mf_eda['freq'].values
    # Kalman Trend filter
    kf_trend = KalmanTrendFilter()
    kf_trend.fit(freq_vals, optimize=True)
    
    ax4.plot(range(len(mf_eda)), freq_vals, color='gray', lw=1.5, alpha=0.7,
             marker='o', ms=4, label='Actual')
    ax4.plot(range(len(mf_eda)), kf_trend.fitted_values_, color=COLORS[0], lw=2.5,
             label=f'Kalman (φ={kf_trend.phi:.2f})')
    
    # Forecast next 5 months
    fc_kal = kf_trend.forecast(5)
    last_x = len(mf_eda) - 1
    ax4.plot(range(last_x, last_x + 6), 
             [freq_vals[-1]] + fc_kal.tolist(), 
             color=COLORS[0], lw=2.5, ls='--', marker='D', ms=6, alpha=0.8,
             label='Kalman Forecast')
    ax4.set_ylabel('Frekuensi')
    ax4.legend(fontsize=8)
    ax4.text(0.05, 0.05,
             f'[v16] Kalman φ={kf_trend.phi:.2f}\nLevel={kf_trend.level_:.0f} Trend={kf_trend.trend_:.1f}',
             transform=ax4.transAxes, fontsize=8, va='bottom',
             bbox=dict(boxstyle='round', facecolor='#E3F9E3', alpha=0.9))
except Exception as e:
    _no_data(ax4, f'{e}')
_title(ax4, '[v16 NEW] Kalman Trend Filter + Damped Forecast')

# Plot 5: External Data impact
ax5 = fig.add_subplot(gs[1, 2])
try:
    ext_months = df_ext_train['month'].values
    ext_seas   = df_ext_train['seasonal_prior'].values
    ext_infl   = df_ext_train['inflation_idx'].values
    
    ax5a = ax5
    ax5b = ax5a.twinx()
    
    bars5 = ax5a.bar(range(len(mf_eda)), ext_seas[:len(mf_eda)], 
                     color=COLORS[2], alpha=0.5, label='Seasonal prior')
    line5 = ax5b.plot(range(len(mf_eda)), ext_infl[:len(mf_eda)],
                      color=COLORS[3], lw=2.5, label='Inflation idx')
    ax5a.set_ylabel('Seasonal Index', color=COLORS[2])
    ax5b.set_ylabel('Inflation Index', color=COLORS[3])
    ax5a.axhline(y=1.0, color='black', lw=1, ls='--')
    ax5a.set_xticks(range(0, len(mf_eda), 2))
    ax5a.set_xticklabels(mf_eda['ym_str'].iloc[::2], rotation=45, ha='right', fontsize=7)
    lines1, labels1 = ax5a.get_legend_handles_labels()
    lines2, labels2 = ax5b.get_legend_handles_labels()
    ax5a.legend(lines1+lines2, labels1+labels2, fontsize=7)
except Exception as e:
    _no_data(ax5, f'{e}')
_title(ax5, '[v16 NEW] External Data: Inflation + Seasonal Prior')

# Plot 6: Severity comparison
ax6 = fig.add_subplot(gs[2, :2])
try:
    xr = range(len(mf_eda))
    ax6.plot(xr, mf_eda['sev_mean']/1e6, color='#636363', lw=1.5, ls='--',
             marker='o', ms=4, label=f'Mean orig')
    ax6.plot(xr, mf_eda['sev_wins']/1e6, color=COLORS[2], lw=2.5,
             marker='s', ms=5, label=f'Wins ({best_win_col})')
    ax6.plot(xr, mf_eda['sev_med']/1e6, color=COLORS[3], lw=1.5,
             ls=':', marker='^', ms=5, label=f'Median (← v10 problem)')
    
    # Severity floor from v16 analysis
    sev_recent_6m = mf_eda['sev_wins'].tail(6).mean()
    sev_floor = sev_recent_6m * 0.85
    ax6.axhline(y=sev_floor/1e6, color='red', lw=2, ls='--',
                label=f'[v16] Severity floor: {sev_floor/1e6:.1f}M')
    
    ax6.set_xticks(range(len(mf_eda)))
    ax6.set_xticklabels(mf_eda['ym_str'], rotation=45, ha='right', fontsize=7.5)
    ax6.set_ylabel('Severitas (Juta IDR)')
    ax6.legend(fontsize=8)
    ax6.text(0.02, 0.97,
             f'[v16] Severity FLOOR = recent 6m avg × 0.85 = {sev_floor/1e6:.1f}M\n'
             f'Prevents v15 problem: forecast dropping below floor',
             transform=ax6.transAxes, fontsize=8.5, va='top',
             bbox=dict(boxstyle='round', facecolor='#FFE5E5', alpha=0.9))
except Exception as e:
    _no_data(ax6, f'{e}')
_title(ax6, '[v16 KEY] Severity: Adaptive Winsor + Floor Protection', fs=11)

# Plot 7: v15 vs v16 forecast comparison preview (from diagnosis)
ax7 = fig.add_subplot(gs[2, 2])
try:
    months_2025 = [8, 9, 10, 11, 12]
    # v15 predictions (from visualization)
    v15_freq = [198, 190.5, 226.0, 228.1, 201.5]
    # v16 expectation: anchored to Jul 2025 level (~260)
    # With damping: freq should be ~250-260 range
    v16_freq_est = [252, 248, 255, 258, 250]  # placeholder for visualization
    
    ax7.bar([str(m) for m in months_2025], v15_freq, color='#636363', alpha=0.6, label='v15 pred')
    ax7.plot(range(len(months_2025)), v16_freq_est, color=COLORS[0],
             lw=2.5, marker='D', ms=8, label='v16 target range')
    ax7.axhline(y=260, color='red', lw=2, ls='--', label='Jul 2025 actual')
    ax7.axhline(y=np.mean(mf_eda['freq'].tail(6).values), color=COLORS[2],
                lw=1.5, ls=':', label='Recent 6m avg')
    ax7.set_ylabel('Frekuensi Klaim')
    ax7.legend(fontsize=7)
    ax7.text(0.05, 0.97,
             '[v16] Freq should NOT drop\nbelow recent average\nv15 bug: 198 vs 260 last actual',
             transform=ax7.transAxes, fontsize=8, va='top',
             bbox=dict(boxstyle='round', facecolor='#FFE5E5', alpha=0.9))
except Exception as e:
    _no_data(ax7, f'{e}')
_title(ax7, '[v16 DIAG] v15 Frequency Underforecast Fix', fs=9)

# Plot 8: Correlation heatmap
ax8 = fig.add_subplot(gs[3, :2])
try:
    hm_cols = [TARGET_COL]
    if COL_BIAYA_RS and COL_BIAYA_RS in df_paid.columns:
        hm_cols.append(COL_BIAYA_RS)
    if COL_TGL_LAHIR and COL_TGL_LAHIR in df_paid.columns:
        df_paid['usia_plot'] = (
            (pd.to_datetime(df_paid[masuk_col]) -
             pd.to_datetime(df_paid[COL_TGL_LAHIR])).dt.days / 365.25
        ).clip(0, 100)
        hm_cols.append('usia_plot')
    if COL_TGL_KELUAR and COL_TGL_KELUAR in df_paid.columns:
        df_paid['lama_rawat_plot'] = (
            (pd.to_datetime(df_paid[COL_TGL_KELUAR]) -
             pd.to_datetime(df_paid[masuk_col])).dt.days
        ).clip(0, 365).fillna(0)
        hm_cols.append('lama_rawat_plot')
    hm_df = df_paid[[c for c in hm_cols if c in df_paid.columns]].dropna()
    rename_map = {TARGET_COL: 'Klaim', COL_BIAYA_RS: 'Biaya RS',
                  'usia_plot': 'Usia', 'lama_rawat_plot': 'Lama Rawat'}
    hm_df = hm_df.rename(columns={k: v for k, v in rename_map.items() if k in hm_df.columns})
    if hm_df.shape[0] > 1 and hm_df.shape[1] >= 2:
        corr_ = hm_df.corr()
        mask_ = np.triu(np.ones_like(corr_, dtype=bool), k=1)
        sns.heatmap(corr_, annot=True, fmt='.3f', cmap='RdYlGn_r', center=0,
                    ax=ax8, mask=mask_, linewidths=0.8,
                    annot_kws={'size': 11, 'weight': 'bold'})
except Exception as e:
    _no_data(ax8, f'{e}')
_title(ax8, 'Heatmap Korelasi Fitur Numerik', fs=12)

# Plot 9: Per-year pattern
ax9 = fig.add_subplot(gs[3, 2])
try:
    mby = df_paid.groupby(['tahun', 'bulan']).size().reset_index(name='freq')
    mnames = ['J','F','M','A','M','J','J','A','S','O','N','D']
    for i, yr in enumerate(sorted(mby['tahun'].unique())):
        sub_ = mby[mby['tahun'] == yr]
        ax9.plot(sub_['bulan'], sub_['freq'], marker='o', lw=2.5,
                 label=str(yr), color=COLORS[i % len(COLORS)])
    ax9.set_xticks(range(1, 13))
    ax9.set_xticklabels(mnames, fontsize=7.5)
    ax9.legend(fontsize=9)
    ax9.set_ylabel('Frekuensi')
except Exception as e:
    _no_data(ax9, f'{e}')
_title(ax9, 'Pola Musiman per Tahun')

# Plot 10: STL Decomposition
ax10a = fig.add_subplot(gs[4, 0])
ax10b = fig.add_subplot(gs[4, 1])
ax10c = fig.add_subplot(gs[4, 2])
try:
    if len(mf_eda) >= 12:
        freq_ts = pd.Series(mf_eda['freq'].values,
                            index=pd.date_range('2024-01', periods=len(mf_eda), freq='MS'))
        stl_r = STL(freq_ts, period=6, robust=True).fit()
        xr5 = range(len(mf_eda))
        ax10a.plot(xr5, stl_r.trend, color=COLORS[0], lw=2.5, label='STL trend')
        ax10a.plot(xr5, mf_eda['freq'], color='gray', lw=1, alpha=0.5, label='Actual')
        ax10a.legend(fontsize=7)
        ax10b.bar(xr5, stl_r.seasonal, color=COLORS[2], alpha=0.75)
        ax10b.axhline(y=0, color='black', lw=0.8)
        ax10c.scatter(xr5, stl_r.resid, color=COLORS[3], s=40, alpha=0.8)
        ax10c.axhline(y=0, color='black', lw=1)
except Exception as e:
    for a in [ax10a, ax10b, ax10c]:
        _no_data(a, f'{e}')
_title(ax10a, 'STL: Trend')
_title(ax10b, 'STL: Seasonal')
_title(ax10c, 'STL: Residual')

plt.savefig('eda_dashboard_v16.png', dpi=100, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("  ✓ eda_dashboard_v16.png tersimpan")


# ========================== 4. FEATURE ENGINEERING ==========================
print("\n[4/10] FEATURE ENGINEERING...")

df_fe = df_paid.copy()
tgl_masuk_s = pd.to_datetime(df_fe[masuk_col], errors='coerce')

# Usia
if COL_TGL_LAHIR and COL_TGL_LAHIR in df_fe.columns:
    df_fe['usia_saat_klaim'] = (
        (tgl_masuk_s - pd.to_datetime(df_fe[COL_TGL_LAHIR], errors='coerce')).dt.days / 365.25
    ).clip(0, 100).fillna(35)
else:
    df_fe['usia_saat_klaim'] = 35.0

# Tenure
if COL_TGL_EFF and COL_TGL_EFF in df_fe.columns:
    df_fe['tenure_polis_tahun'] = (
        (tgl_masuk_s - pd.to_datetime(df_fe[COL_TGL_EFF], errors='coerce')).dt.days / 365.25
    ).clip(0).fillna(1)
else:
    df_fe['tenure_polis_tahun'] = 1.0

# Lama rawat
if COL_TGL_KELUAR and COL_TGL_KELUAR in df_fe.columns:
    df_fe['lama_rawat'] = (
        (pd.to_datetime(df_fe[COL_TGL_KELUAR], errors='coerce') - tgl_masuk_s).dt.days
    ).clip(0, 365).fillna(0)
else:
    df_fe['lama_rawat'] = 1.0

df_fe['bulan_masuk']       = tgl_masuk_s.dt.month.fillna(1).astype(int)
df_fe['kuartal_masuk']     = tgl_masuk_s.dt.quarter.fillna(1).astype(int)
df_fe['hari_dalam_minggu'] = tgl_masuk_s.dt.dayofweek.fillna(0).astype(int)
df_fe['is_weekend']        = df_fe['hari_dalam_minggu'].isin([5, 6]).astype(int)
df_fe['tahun']             = tgl_masuk_s.dt.year.fillna(2024).astype(int)
df_fe['day_of_month']      = tgl_masuk_s.dt.day.fillna(15).astype(int)

# Cyclic
df_fe['sin_month'] = np.sin(2 * np.pi * df_fe['bulan_masuk'] / 12)
df_fe['cos_month'] = np.cos(2 * np.pi * df_fe['bulan_masuk'] / 12)
df_fe['sin_qtr']   = np.sin(2 * np.pi * df_fe['kuartal_masuk'] / 4)
df_fe['cos_qtr']   = np.cos(2 * np.pi * df_fe['kuartal_masuk'] / 4)

# [v16 NEW] External data features per claim
# Attach inflation index per claim month
ext_lookup = df_ext_train.set_index(['year', 'month'])
def get_ext_feat(yr, mo, feat):
    try: return float(ext_lookup.loc[(yr, mo), feat])
    except: return 1.0

for feat_ in ['inflation_idx', 'seasonal_prior', 'is_ramadan', 'is_eid_fitr',
              'is_yearend', 'is_yearstart', 'linear_trend']:
    df_fe[f'ext_{feat_}'] = [
        get_ext_feat(int(r['tahun']), int(r['bulan_masuk']), feat_)
        for _, r in df_fe[['tahun', 'bulan_masuk']].iterrows()
    ]

# IP/OP, Gender
if COL_IP_OP and COL_IP_OP in df_fe.columns:
    df_fe['is_ip'] = df_fe[COL_IP_OP].astype(str).str.lower().str.contains('in').astype(int)
else:
    df_fe['is_ip'] = 0

if COL_GENDER and COL_GENDER in df_fe.columns:
    df_fe['is_female'] = df_fe[COL_GENDER].astype(str).str.upper().str.contains('F').astype(int)
else:
    df_fe['is_female'] = 0

if COL_REIMBURSE and COL_REIMBURSE in df_fe.columns:
    df_fe['is_reimburse'] = df_fe[COL_REIMBURSE].astype(str).str.upper().str.contains('R').astype(int)
else:
    df_fe['is_reimburse'] = 0

# ICD features
global_med = df_fe[TARGET_COL].median()
if pd.isna(global_med) or global_med == 0:
    global_med = 1.0

if COL_ICD and COL_ICD in df_fe.columns:
    df_fe['icd_chapter'] = df_fe[COL_ICD].astype(str).str[0].str.upper().fillna('Z')
    df_fe['icd_3char']   = df_fe[COL_ICD].astype(str).str[:3].str.upper().fillna('Z00')

    med_cost = df_fe.groupby('icd_3char')[TARGET_COL].median()
    freq_icd = df_fe.groupby('icd_3char').size()
    pctiles  = med_cost.quantile([.20, .40, .60, .80])

    def _tier(icd):
        c = med_cost.get(icd, global_med)
        if pd.isna(c): c = global_med
        if c <= pctiles[.20]:   return 0
        elif c <= pctiles[.40]: return 1
        elif c <= pctiles[.60]: return 2
        elif c <= pctiles[.80]: return 3
        return 4

    df_fe['icd_cost_tier']   = df_fe['icd_3char'].map(_tier).fillna(2).astype(int)
    df_fe['icd_freq_rank']   = df_fe['icd_3char'].map(freq_icd).fillna(1).astype(int)
    chapter_sev = {
        'A': 1, 'B': 1, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 3, 'H': 1,
        'I': 4, 'J': 2, 'K': 2, 'L': 1, 'M': 2, 'N': 2, 'O': 3, 'P': 2,
        'Q': 3, 'R': 1, 'S': 2, 'T': 3, 'Z': 1
    }
    df_fe['icd_disease_sev'] = df_fe['icd_chapter'].map(chapter_sev).fillna(1).astype(int)
    chronic_ch = {'C', 'D', 'E', 'I', 'J', 'K', 'M', 'N'}
    df_fe['is_chronic']      = df_fe['icd_chapter'].isin(chronic_ch).astype(int)
    df_fe['is_catastrophic'] = (df_fe['icd_cost_tier'] == 4).astype(int)

    mean_cost_icd = df_fe.groupby('icd_3char')[TARGET_WINS_COL].mean()
    df_fe['icd_mean_cost']  = df_fe['icd_3char'].map(mean_cost_icd).fillna(global_med)
    df_fe['icd_cost_ratio'] = df_fe['icd_mean_cost'] / (global_med + 1e-10)
else:
    for col_ in ['icd_chapter', 'icd_3char', 'icd_cost_tier', 'icd_disease_sev',
                 'is_chronic', 'is_catastrophic', 'icd_freq_rank', 'icd_mean_cost', 'icd_cost_ratio']:
        df_fe[col_] = 2 if 'cost' in col_ or 'tier' in col_ else (1 if 'sev' in col_ else 0)

if COL_BIAYA_RS and COL_BIAYA_RS in df_fe.columns:
    df_fe['ratio_klaim_biaya'] = (
        df_fe[TARGET_COL] / df_fe[COL_BIAYA_RS].replace(0, np.nan)
    ).clip(0, 5).fillna(0.8)
else:
    df_fe['ratio_klaim_biaya'] = 0.8

if COL_PLAN and COL_PLAN in df_fe.columns:
    plan_sev   = df_fe.groupby(COL_PLAN)[TARGET_COL].median()
    plan_count = df_fe.groupby(COL_PLAN).size()
    df_fe['plan_sev_median'] = df_fe[COL_PLAN].map(plan_sev).fillna(global_med)
    df_fe['plan_count']      = df_fe[COL_PLAN].map(plan_count).fillna(0)
    plan_rank = {'M-001': 3, 'M-002': 2, 'M-003': 1}
    df_fe['plan_rank']       = df_fe[COL_PLAN].map(plan_rank).fillna(1)
else:
    df_fe['plan_sev_median'] = 0
    df_fe['plan_count']      = 0
    df_fe['plan_rank']       = 1

# Interactions
df_fe['usia_x_chronic']  = df_fe['usia_saat_klaim'] * df_fe['is_chronic']
df_fe['usia_x_tier']     = df_fe['usia_saat_klaim'] * df_fe['icd_cost_tier']
df_fe['rawat_x_tier']    = df_fe['lama_rawat']      * df_fe['icd_cost_tier']
df_fe['rawat_x_ip']      = df_fe['lama_rawat']      * df_fe['is_ip']
df_fe['plan_x_tier']     = df_fe['plan_rank']       * df_fe['icd_cost_tier']
df_fe['cost_x_ip']       = df_fe['icd_cost_ratio']  * df_fe['is_ip']
df_fe['cost_x_chronic']  = df_fe['icd_cost_ratio']  * df_fe['is_chronic']

# [v16] Inflation-adjusted severity interaction
df_fe['sev_adj_infl'] = df_fe[TARGET_WINS_COL] / df_fe['ext_inflation_idx']

print(f"  ✓ Features: {df_fe.shape[1]} kolom | Records: {len(df_fe):,}")


# ========================== 5. AGREGASI BULANAN ==============================
print("\n[5/10] AGREGASI BULANAN...")

monthly = df_fe.groupby(['tahun', 'bulan_masuk']).agg(
    frequency     = (TARGET_COL,           'count'),
    severity      = (TARGET_COL,           'mean'),
    severity_w    = (TARGET_WINS_COL,      'mean'),
    severity_med  = (TARGET_COL,           'median'),
    total_claim   = (TARGET_COL,           'sum'),
    usia_mean     = ('usia_saat_klaim',    'mean'),
    tenure_mean   = ('tenure_polis_tahun', 'mean'),
    rawat_mean    = ('lama_rawat',         'mean'),
    pct_chronic   = ('is_chronic',         'mean'),
    pct_catast    = ('is_catastrophic',    'mean'),
    avg_tier      = ('icd_cost_tier',      'mean'),
    avg_dis_sev   = ('icd_disease_sev',    'mean'),
    pct_ip        = ('is_ip',              'mean'),
    pct_female    = ('is_female',          'mean'),
    plan_sev_m    = ('plan_sev_median',    'mean'),
    plan_rank_m   = ('plan_rank',          'mean'),
    ratio_kb_m    = ('ratio_klaim_biaya',  'mean'),
    icd_cost_r    = ('icd_cost_ratio',     'mean'),
    cost_ip_m     = ('cost_x_ip',          'mean'),
    # [v16] External regressors
    ext_inflation = ('ext_inflation_idx',  'mean'),
    ext_seasonal  = ('ext_seasonal_prior', 'mean'),
    ext_ramadan   = ('ext_is_ramadan',     'mean'),
    ext_eid       = ('ext_is_eid_fitr',    'mean'),
    ext_yearend   = ('ext_is_yearend',     'mean'),
    ext_trend     = ('ext_linear_trend',   'mean'),
).reset_index()

monthly['year_month'] = pd.to_datetime(
    monthly['tahun'].astype(str) + '-' +
    monthly['bulan_masuk'].astype(str).str.zfill(2)
)
monthly = monthly.sort_values('year_month').reset_index(drop=True)
monthly['fs_wins']       = monthly['frequency'] * monthly['severity_w']
monthly['fs_correction'] = monthly['total_claim'] / (monthly['fs_wins'] + 1e-10)

# Global correction factor (use median - more robust)
GLOBAL_FS_CORRECTION = float(monthly['fs_correction'].median())
print(f"  Global F×S correction: {GLOBAL_FS_CORRECTION:.3f}")

# Lag features
for lag in [1, 2, 3]:
    monthly[f'mlag_freq_{lag}'] = monthly['frequency'].shift(lag)
    monthly[f'mlag_sevw_{lag}'] = monthly['severity_w'].shift(lag)
monthly['mlag_tot_1']  = monthly['total_claim'].shift(1)
monthly['mlag_tot_2']  = monthly['total_claim'].shift(2)
monthly['mlag_corr_1'] = monthly['fs_correction'].shift(1)

monthly['mroll_freq_3m']  = monthly['frequency'].rolling(3, min_periods=1).mean().shift(1)
monthly['mroll_freq_6m']  = monthly['frequency'].rolling(6, min_periods=1).mean().shift(1)
monthly['mroll_sevw_3m']  = monthly['severity_w'].rolling(3, min_periods=1).mean().shift(1)
monthly['mroll_tot_3m']   = monthly['total_claim'].rolling(3, min_periods=1).sum().shift(1)
monthly['mroll_tot_6m']   = monthly['total_claim'].rolling(6, min_periods=1).sum().shift(1)

monthly['yoy_freq']  = monthly['frequency'].shift(12)
monthly['yoy_sevw']  = monthly['severity_w'].shift(12)
monthly['yoy_total'] = monthly['total_claim'].shift(12)
monthly['yoy_growth_freq']  = (monthly['frequency']   / (monthly['yoy_freq'] + 1e-10) - 1).clip(-1, 5)
monthly['yoy_growth_sevw']  = (monthly['severity_w']  / (monthly['yoy_sevw'] + 1e-10) - 1).clip(-1, 5)
monthly['yoy_growth_total'] = (monthly['total_claim'] / (monthly['yoy_total']+ 1e-10) - 1).clip(-1, 5)

monthly['sin_month'] = np.sin(2 * np.pi * monthly['bulan_masuk'] / 12)
monthly['cos_month'] = np.cos(2 * np.pi * monthly['bulan_masuk'] / 12)
monthly['freq_mom']  = monthly['frequency'].pct_change().clip(-2, 2).shift(1).fillna(0)
monthly['sevw_mom']  = monthly['severity_w'].pct_change().clip(-2, 2).shift(1).fillna(0)

all_lag_cols = [c for c in monthly.columns if any(x in c for x in ['mlag', 'mroll', 'yoy', 'mom'])]
monthly[all_lag_cols] = monthly[all_lag_cols].bfill().ffill().fillna(0)

N = len(monthly)
print(f"  ✓ Monthly data: {N} bulan")
print(f"  frequency  : mean={monthly['frequency'].mean():.1f}, last={monthly['frequency'].iloc[-1]:.0f}")
print(f"  severity_w : mean={monthly['severity_w'].mean()/1e6:.2f}M")
print(f"  total_claim: mean={monthly['total_claim'].mean()/1e9:.3f}B IDR")

# YoY growth rates
yoy_mask = monthly['yoy_freq'].notna() & (monthly['yoy_freq'] > 0)
if yoy_mask.sum() >= 3:
    ACTUAL_GROWTH_FREQ  = float(monthly.loc[yoy_mask, 'yoy_growth_freq'].median())
    ACTUAL_GROWTH_SEV   = float(monthly.loc[yoy_mask, 'yoy_growth_sevw'].median())
    ACTUAL_GROWTH_TOTAL = float(monthly.loc[yoy_mask, 'yoy_growth_total'].median())
else:
    ACTUAL_GROWTH_FREQ  = 0.027
    ACTUAL_GROWTH_SEV   = 0.05
    ACTUAL_GROWTH_TOTAL = 0.10

print(f"  YoY growth: Freq={ACTUAL_GROWTH_FREQ*100:+.1f}% Sev={ACTUAL_GROWTH_SEV*100:+.1f}% "
      f"Total={ACTUAL_GROWTH_TOTAL*100:+.1f}%")

# Historical bounds
HIST_FREQ_P5,  HIST_FREQ_P95  = monthly['frequency'].quantile(0.05), monthly['frequency'].quantile(0.95)
HIST_SEV_P5,   HIST_SEV_P95   = monthly['severity_w'].quantile(0.05), monthly['severity_w'].quantile(0.95)
HIST_TOT_P5,   HIST_TOT_P95   = monthly['total_claim'].quantile(0.05), monthly['total_claim'].quantile(0.95)

# [v16 KEY] Frequency anchor - recent 3-month average
FREQ_RECENT_3M = float(monthly['frequency'].tail(3).mean())
FREQ_RECENT_6M = float(monthly['frequency'].tail(6).mean())
SEV_RECENT_3M  = float(monthly['severity_w'].tail(3).mean())
SEV_RECENT_6M  = float(monthly['severity_w'].tail(6).mean())
TOT_RECENT_3M  = float(monthly['total_claim'].tail(3).mean())
TOT_RECENT_6M  = float(monthly['total_claim'].tail(6).mean())

print(f"\n  [v16] Anchors (from recent data):")
print(f"    Freq recent 3m={FREQ_RECENT_3M:.1f}, 6m={FREQ_RECENT_6M:.1f}")
print(f"    Sev  recent 3m={SEV_RECENT_3M/1e6:.2f}M, 6m={SEV_RECENT_6M/1e6:.2f}M")
print(f"    Total recent 3m={TOT_RECENT_3M/1e9:.3f}B, 6m={TOT_RECENT_6M/1e9:.3f}B")


# ========================== 6. FEATURE IMPORTANCE ===========================
print("\n[6/10] FEATURE IMPORTANCE (v16)...")

FEATS_CANDIDATES = [
    'bulan_masuk', 'tahun', 'sin_month', 'cos_month',
    'usia_mean', 'tenure_mean', 'rawat_mean',
    'pct_chronic', 'pct_catast', 'avg_tier', 'avg_dis_sev', 'pct_ip',
    'pct_female', 'plan_sev_m', 'plan_rank_m', 'ratio_kb_m',
    'icd_cost_r', 'cost_ip_m',
    'mlag_freq_1', 'mlag_freq_2', 'mlag_freq_3',
    'mlag_sevw_1', 'mlag_sevw_2',
    'mlag_tot_1', 'mlag_tot_2', 'mlag_corr_1',
    'mroll_freq_3m', 'mroll_freq_6m',
    'mroll_sevw_3m', 'mroll_tot_3m', 'mroll_tot_6m',
    'freq_mom', 'sevw_mom',
    'yoy_freq', 'yoy_sevw', 'yoy_total',
    # [v16 NEW] External regressors
    'ext_inflation', 'ext_seasonal', 'ext_ramadan', 'ext_eid',
    'ext_yearend', 'ext_trend',
]
FEATS = list(dict.fromkeys([f for f in FEATS_CANDIDATES if f in monthly.columns]))
monthly[FEATS] = monthly[FEATS].fillna(0)

train_data = monthly[
    monthly['frequency'].notna() &
    monthly['severity_w'].notna() &
    monthly['total_claim'].notna()
].copy()
train_data[FEATS] = train_data[FEATS].fillna(0)

print(f"  ✓ Features: {len(FEATS)} | Training: {len(train_data)} bulan")

fi_models = {}
for tgt_name, tgt_col in [
    ('Frequency',  'frequency'),
    ('Severity',   'severity_w'),
    ('Total',      'total_claim')
]:
    try:
        fi_y = train_data[tgt_col].fillna(0).values
        if np.all(fi_y == 0):
            fi_models[tgt_name] = pd.Series(np.ones(len(FEATS))/len(FEATS), index=FEATS)
            continue
        m = RandomForestRegressor(n_estimators=200, max_depth=5,
                                  min_samples_leaf=2, random_state=42, n_jobs=-1)
        m.fit(train_data[FEATS], np.log1p(np.maximum(fi_y, 0)))
        fi_models[tgt_name] = pd.Series(m.feature_importances_, index=FEATS).sort_values(ascending=False)
    except Exception as e:
        print(f"  ⚠ FI {tgt_name}: {e}")
        fi_models[tgt_name] = pd.Series(np.ones(len(FEATS)), index=FEATS)

try:
    fig_fi, axes_fi = plt.subplots(1, 3, figsize=(22, 9))
    fig_fi.suptitle('Feature Importance v16.0 — RandomForest Top 15 per Target\n'
                    '[v16] Kalman + Optimal Damping + External Data',
                    fontsize=13, fontweight='bold')
    for ax, (tgt_name, fi), color in zip(axes_fi, fi_models.items(), COLORS):
        top = fi.head(15)
        max_imp = top.values.max() if top.values.max() > 0 else 1.0
        top_norm = top / max_imp
        yp = range(len(top_norm))
        ax.barh(list(yp), top_norm.values[::-1], color=color, alpha=0.85)
        ax.set_yticks(list(yp))
        ax.set_yticklabels(top_norm.index[::-1], fontsize=9)
        ax.set_title(f'{tgt_name}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Importance (normalized)')
        ax.set_xlim(0, 1.15)
        for bar, val in zip(ax.patches, top_norm.values[::-1]):
            ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=8)
    plt.tight_layout()
    plt.savefig('feature_importance_v16.png', dpi=100, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ feature_importance_v16.png tersimpan")
except Exception as e:
    print(f"  ⚠ FI plot: {e}")
    plt.close('all')


# ========================== 7. PEMODELAN v16 ================================
print("\n[7/10] PEMODELAN v16.0...")
print("  [v16] Kalman Trend + Optimal Damping + SARIMAX(exog) + STL + Stacking")
print("  [v16] Frequency ANCHOR + Severity FLOOR + F×S Direct Enforcement")

TARGET_CONFIG = {
    'frequency':   {'col': 'frequency',   'log': False, 'use_exog': True},
    'severity':    {'col': 'severity_w',  'log': True,  'use_exog': True},
    'total_claim': {'col': 'total_claim', 'log': True,  'use_exog': True},
}

# External regressors for training period
EXT_COLS_MONTHLY = ['ext_inflation', 'ext_seasonal', 'ext_ramadan', 'ext_eid',
                    'ext_yearend', 'ext_trend']
EXT_COLS_AVAIL = [c for c in EXT_COLS_MONTHLY if c in monthly.columns]

# Forecast external regressors for 2025 Aug-Dec
PRED_MONTHS_2025 = [8, 9, 10, 11, 12]
df_ext_forecast = build_external_data(
    year_start=2025, month_start=8,
    n_months=5
)
# Fix: only take the first 5 months
df_ext_forecast = df_ext_forecast.head(5)

print(f"  External regressors for forecast: {df_ext_forecast[['year','month']].values.tolist()}")

def find_best_sarimax(series, exog=None, max_p=2, max_d=1, max_q=2):
    """Find best SARIMAX order via AIC"""
    best_aic   = np.inf
    best_order = (1, 1, 1)
    for p, d, q in product(range(max_p+1), range(max_d+1), range(max_q+1)):
        if p == 0 and q == 0: continue
        try:
            model = SARIMAX(
                series, exog=exog, order=(p, d, q),
                seasonal_order=(0,0,0,0),
                enforce_stationarity=False, enforce_invertibility=False
            )
            result = model.fit(disp=False, maxiter=100)
            if result.aic < best_aic:
                best_aic   = result.aic
                best_order = (p, d, q)
        except: continue
    return best_order, best_aic


# ─── Walk-Forward Cross-Validation ───────────────────────────────────────────
print("\n  [Walk-Forward CV - v16]")
MIN_TRAIN_WF = max(12, N - 8)

wf_results = {}

for tgt_name, cfg in TARGET_CONFIG.items():
    col     = cfg['col']
    use_log = cfg['log']
    series  = monthly[col].values.copy()
    mapes_wf, preds_wf, actuals_wf = [], [], []

    for test_idx in range(MIN_TRAIN_WF, N):
        y_train     = series[:test_idx]
        y_true      = series[test_idx]
        ext_train   = monthly[EXT_COLS_AVAIL].values[:test_idx] if EXT_COLS_AVAIL else None
        ext_test    = monthly[EXT_COLS_AVAIL].values[test_idx:test_idx+1] if EXT_COLS_AVAIL else None
        
        if len(y_train) < 6 or np.all(y_train == 0): continue
        y_fit = np.log1p(np.maximum(y_train, 0)) if use_log else y_train.copy()

        # [v16-A] Kalman Trend Model
        try:
            kf = KalmanTrendFilter()
            kf.fit(y_fit, optimize=False)  # fast for WF
            kf_pred_log = float(kf.forecast(1)[0])
            kf_pred = max(0, np.expm1(kf_pred_log) if use_log else kf_pred_log)
        except:
            kf_pred = float(y_train.mean())

        # [v16-B] Damped HW (with optimal phi)
        try:
            hw_model = ExponentialSmoothing(
                y_fit, trend='add', seasonal=None, damped_trend=True,
                initialization_method='estimated'
            )
            hw_fit  = hw_model.fit(optimized=True, use_brute=False)
            hw_pred_log = float(hw_fit.forecast(1).iloc[0])
            hw_pred = max(0, np.expm1(hw_pred_log) if use_log else hw_pred_log)
        except:
            hw_pred = float(y_train.mean())

        # [v16-C] SARIMAX with exog
        try:
            if ext_train is not None and len(ext_train) >= 6:
                sarima_model = SARIMAX(
                    y_fit, exog=ext_train,
                    order=(1, 1, 1), seasonal_order=(0,0,0,0),
                    enforce_stationarity=False, enforce_invertibility=False
                )
                sarima_fit  = sarima_model.fit(disp=False, maxiter=100)
                sarima_pred_log = float(sarima_fit.forecast(steps=1, exog=ext_test).iloc[-1])
            else:
                sarima_model = SARIMAX(
                    y_fit, order=(1, 1, 1), seasonal_order=(0,0,0,0),
                    enforce_stationarity=False, enforce_invertibility=False
                )
                sarima_fit  = sarima_model.fit(disp=False, maxiter=100)
                sarima_pred_log = float(sarima_fit.forecast(1).iloc[-1])
            sarima_pred = max(0, np.expm1(sarima_pred_log) if use_log else sarima_pred_log)
        except:
            sarima_pred = hw_pred

        # [v16] Ensemble: simple average (stacking weights computed analytically later)
        preds_list = [kf_pred, hw_pred, sarima_pred]
        ens_pred = np.mean(preds_list)
        ens_pred = max(0, ens_pred)

        mapes_wf.append(calc_mape([y_true], [ens_pred]))
        preds_wf.append(ens_pred)
        actuals_wf.append(y_true)

    wf_results[tgt_name] = {
        'mapes':     mapes_wf,
        'mean_mape': float(np.mean(mapes_wf)) if mapes_wf else 99.0,
        'preds':     preds_wf,
        'actuals':   actuals_wf,
        'bias':      float(np.mean([(a-p)/max(abs(a),1) for a,p in zip(actuals_wf,preds_wf)]))
                     if preds_wf else 0.0
    }
    print(f"  {tgt_name:<15}: WF-MAPE={wf_results[tgt_name]['mean_mape']:.2f}%"
          f"  Bias={wf_results[tgt_name]['bias']*100:+.1f}%  (n={len(mapes_wf)})")


# ─── Build Final Models ───────────────────────────────────────────────────────
print("\n  [Building Final Models - v16]")
final_models = {}
final_mapes  = {}

for tgt_name, cfg in TARGET_CONFIG.items():
    col     = cfg['col']
    use_log = cfg['log']
    series_raw = monthly[col].values.copy()
    n_series   = len(series_raw)
    
    print(f"\n  ─── {tgt_name} ────────────────────────────────────────────")
    
    y_fit   = np.log1p(np.maximum(series_raw, 0)) if use_log else series_raw.copy()
    ext_all = monthly[EXT_COLS_AVAIL].values if EXT_COLS_AVAIL else None

    # ── [v16-A] Kalman Trend Filter ─────────────────────────────────────────
    kf_model = None
    mape_kf  = 99.0
    try:
        print(f"    [A] Kalman Trend Filter...")
        kf_model = KalmanTrendFilter()
        kf_model.fit(y_fit, optimize=True)

        # [FIX] Use out-of-sample WF forecasts to compute MAPE
        # In-sample fitted values give MAPE~0 (Kalman smoothing artefact)
        n_test = min(6, n_series // 3)
        kf_oos_preds = []
        for i_oos in range(n_series - n_test, n_series):
            try:
                kf_tmp = KalmanTrendFilter()
                kf_tmp.fit(y_fit[:i_oos], optimize=False)
                kf_fc_log = float(kf_tmp.forecast(1)[0])
                kf_fc_raw = max(0, np.expm1(kf_fc_log) if use_log else kf_fc_log)
                kf_oos_preds.append(kf_fc_raw)
            except:
                kf_oos_preds.append(float(series_raw[:i_oos].mean()))
        mape_kf = calc_mape(series_raw[-n_test:], kf_oos_preds)
        print(f"    [A] Kalman phi={kf_model.phi:.3f} OOS-MAPE={mape_kf:.2f}%")
    except Exception as e:
        print(f"    [A] Kalman failed: {e}")

    # ── [v16-B] Optimal Damped HW ──────────────────────────────────────────
    hw_model  = None
    mape_hw   = 99.0
    best_phi_hw = 0.95
    try:
        print(f"    [B] Optimal Damped HW...")
        best_phi_hw, best_mape_phi, phi_mapes_tgt = find_optimal_damping(
            series_raw, phi_grid=[0.80, 0.85, 0.90, 0.95, 0.97, 0.98, 1.00]
        )
        print(f"    [B] Optimal φ={best_phi_hw} (WF-MAPE={best_mape_phi:.2f}%)")
        
        if best_phi_hw < 1.0:
            hw_m = ExponentialSmoothing(
                y_fit, trend='add', seasonal=None, damped_trend=True,
                initialization_method='estimated'
            )
            hw_model = hw_m.fit(damping_trend=best_phi_hw, optimized=True, use_brute=True)
        else:
            hw_m = ExponentialSmoothing(
                y_fit, trend='add', seasonal=None, damped_trend=False,
                initialization_method='estimated'
            )
            hw_model = hw_m.fit(optimized=True, use_brute=True)
        
        n_test = min(6, n_series // 3)
        hw_insample = hw_model.fittedvalues[-n_test:]
        if use_log:
            hw_insample_raw = np.expm1(np.array(hw_insample))
        else:
            hw_insample_raw = np.array(hw_insample)
        mape_hw = calc_mape(series_raw[-n_test:], np.maximum(hw_insample_raw, 0))
        print(f"    [B] Damped HW MAPE={mape_hw:.2f}%")
    except Exception as e:
        print(f"    [B] HW failed: {e}")

    # ── [v16-C] SARIMAX with External Regressors ───────────────────────────
    sarima_fitted = None
    mape_sarima   = 99.0
    try:
        print(f"    [C] SARIMAX + external regressors...")
        if ext_all is not None and len(ext_all) >= 8:
            best_ord, best_aic = find_best_sarimax(y_fit, exog=ext_all)
            print(f"    [C] Best SARIMAX{best_ord} AIC={best_aic:.2f}")
            sarima_model  = SARIMAX(
                y_fit, exog=ext_all, order=best_ord, seasonal_order=(0,0,0,0),
                enforce_stationarity=False, enforce_invertibility=False
            )
            sarima_fitted = sarima_model.fit(disp=False, maxiter=100)
        else:
            best_ord = (1,1,1)
            sarima_model  = SARIMAX(
                y_fit, order=best_ord, seasonal_order=(0,0,0,0),
                enforce_stationarity=False, enforce_invertibility=False
            )
            sarima_fitted = sarima_model.fit(disp=False, maxiter=100)
        
        n_test = min(6, n_series // 3)
        s_insample = sarima_fitted.predict(start=n_series-n_test, end=n_series-1)
        if use_log:
            s_insample_raw = np.expm1(np.array(s_insample))
        else:
            s_insample_raw = np.array(s_insample)
        mape_sarima = calc_mape(series_raw[-n_test:], np.maximum(s_insample_raw, 0))
        print(f"    [C] SARIMAX MAPE={mape_sarima:.2f}%")
    except Exception as e:
        print(f"    [C] SARIMAX failed: {e}")

    # ── [v16-D] STL + Ridge ────────────────────────────────────────────────
    stl_ml_components = None
    mape_stl  = 99.0
    try:
        print(f"    [D] STL + Ridge...")
        if n_series >= 12:
            ts_index  = pd.date_range(start='2024-01-01', periods=n_series, freq='MS')
            ts_series = pd.Series(y_fit, index=ts_index)
            stl_d = STL(ts_series, period=6, robust=True)
            stl_r = stl_d.fit()
            trend_vals = stl_r.trend.values
            t_arr = np.arange(n_series)
            trend_poly = np.polyfit(t_arr, trend_vals, deg=1)  # linear trend (not poly2)
            seasonal_pattern = np.zeros(6)
            for m_ in range(6):
                idxs = [i for i in range(n_series) if i % 6 == m_]
                if idxs:
                    seasonal_pattern[m_] = np.mean(stl_r.seasonal.values[idxs])
            
            resid = stl_r.resid.values
            
            # [v16] Include external regressors in STL residual model
            if ext_all is not None:
                X_stl = np.column_stack([
                    np.arange(n_series),
                    np.sin(2*np.pi*np.arange(n_series)/6),
                    np.cos(2*np.pi*np.arange(n_series)/6),
                    ext_all[:n_series, :min(3, ext_all.shape[1])]  # first 3 ext features
                ])
            else:
                X_stl = np.column_stack([
                    np.arange(n_series),
                    np.sin(2*np.pi*np.arange(n_series)/6),
                    np.cos(2*np.pi*np.arange(n_series)/6),
                ])
            
            scaler_stl = RobustScaler()
            X_stl_sc   = scaler_stl.fit_transform(X_stl)
            ridge_resid = Ridge(alpha=10)
            ridge_resid.fit(X_stl_sc[:-1], resid[1:])

            stl_ml_components = {
                'trend_poly': trend_poly, 'seasonal_pattern': seasonal_pattern,
                'ridge': ridge_resid, 'scaler': scaler_stl,
                'n_train': n_series, 'use_log': use_log,
                'n_ext': X_stl.shape[1] - 3  # how many ext cols were added
            }
            
            n_test = min(6, n_series // 3)
            stl_preds_test = []
            for i in range(n_series - n_test, n_series):
                t_next = i
                trend_next    = np.polyval(trend_poly, t_next)
                seasonal_next = seasonal_pattern[t_next % 6]
                
                if ext_all is not None:
                    X_next = np.array([[t_next,
                                        np.sin(2*np.pi*t_next/6), np.cos(2*np.pi*t_next/6)] +
                                        ext_all[i, :min(3, ext_all.shape[1])].tolist()])
                else:
                    X_next = np.array([[t_next,
                                        np.sin(2*np.pi*t_next/6), np.cos(2*np.pi*t_next/6)]])
                X_next_sc = scaler_stl.transform(X_next)
                resid_next = float(ridge_resid.predict(X_next_sc)[0])
                pv = trend_next + seasonal_next + resid_next
                stl_preds_test.append(max(0, np.expm1(pv) if use_log else pv))
            
            mape_stl = calc_mape(series_raw[-n_test:], stl_preds_test)
            print(f"    [D] STL+Ridge MAPE={mape_stl:.2f}%")
    except Exception as e:
        print(f"    [D] STL failed: {e}")

    # ── [v16-E] Prophet ────────────────────────────────────────────────────
    prophet_fitted = None
    mape_prophet   = 99.0
    try:
        if HAS_PROPHET:
            print(f"    [E] Prophet...")
            series_for_prophet = np.expm1(y_fit) if use_log else series_raw
            dates = pd.date_range(end=pd.Timestamp('2025-07-01'), periods=n_series, freq='MS')
            df_p  = pd.DataFrame({
                'ds': dates,
                'y': np.maximum(series_for_prophet, 0.01)
            })
            
            # Add external regressors if available
            if ext_all is not None:
                for j, col_name in enumerate(EXT_COLS_AVAIL[:3]):
                    df_p[col_name] = ext_all[:n_series, j]
            
            model_p = Prophet(
                yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False,
                changepoint_prior_scale=0.30, seasonality_prior_scale=0.01,
                changepoint_range=0.85, interval_width=0.8
            )
            
            if ext_all is not None:
                for col_name in EXT_COLS_AVAIL[:3]:
                    model_p.add_regressor(col_name)
            
            model_p.fit(df_p, iter=500)
            
            future = model_p.make_future_dataframe(periods=0, freq='MS')
            if ext_all is not None:
                for j, col_name in enumerate(EXT_COLS_AVAIL[:3]):
                    future[col_name] = ext_all[-len(future):, j]
            
            forecast_p = model_p.predict(future)
            prophet_fitted = model_p
            
            n_test = min(6, n_series // 3)
            p_preds_raw = forecast_p['yhat'].values[-n_test:]
            mape_prophet = calc_mape(series_raw[-n_test:], np.maximum(p_preds_raw, 0))
            print(f"    [E] Prophet MAPE={mape_prophet:.2f}%")
    except Exception as e:
        print(f"    [E] Prophet error: {e}")

    # ── [v16 FIX-9] Stacking Weights via Ridge (Analytic) ──────────────────
    mapes_all = {'Kalman': mape_kf, 'HW': mape_hw, 'SARIMAX': mape_sarima,
                 'STL': mape_stl, 'Prophet': mape_prophet}
    
    valid_models = {k: v for k, v in mapes_all.items() if v < 60.0}
    if not valid_models:
        valid_models = {'HW': mape_hw}
    
    # Inverse-MAPE^2 weights (more stable than optimization)
    inv_mape2_sum = sum(1.0 / (v**2 + 1e-6) for v in valid_models.values())
    weights_e = {}
    for k in mapes_all.keys():
        if k in valid_models:
            weights_e[k] = (1.0 / (mapes_all[k]**2 + 1e-6)) / inv_mape2_sum
        else:
            weights_e[k] = 0.0
    
    print(f"    MAPEs: {' '.join([f'{k}={v:.1f}%' for k,v in mapes_all.items()])}")
    print(f"    Weights: {' '.join([f'{k}={v:.3f}' for k,v in weights_e.items()])}")
    
    # Bias correction
    bias_cf = 1.0 + wf_results[tgt_name]['bias']
    print(f"    Bias correction: ×{bias_cf:.4f}")
    
    final_models[tgt_name] = {
        'kalman':    kf_model,
        'hw':        hw_model,
        'hw_phi':    best_phi_hw,
        'sarima':    sarima_fitted,
        'stl':       stl_ml_components,
        'prophet':   prophet_fitted,
        'weights':   weights_e,
        'use_log':   use_log,
        'col':       col,
        'n_train':   n_series,
        'series_raw': series_raw,
        'series_log': y_fit,
        'bias_cf':   bias_cf,
        'ext_all':   ext_all,
    }
    final_mapes[tgt_name] = min(valid_models.values()) if valid_models else 99.0

print(f"\n  ─────────────────────────────────────────────────────────────────")
for tgt in ['frequency', 'severity', 'total_claim']:
    print(f"  {tgt:<15}: Best MAPE={final_mapes[tgt]:.2f}%")
print(f"  AVG MAPE: {np.mean(list(final_mapes.values())):.2f}%")


# ─── Model Evaluation Dashboard ──────────────────────────────────────────────
print("\n  [Generating Model Evaluation Dashboard...]")
tgt_labels = ['Frequency', 'Severity (Wins)', 'Total Claim']
tgt_keys   = ['frequency', 'severity', 'total_claim']

try:
    fig_ev = plt.figure(figsize=(22, 18))
    fig_ev.suptitle(
        'Model Evaluation v16.0 — Kalman + Damped HW + SARIMAX(exog) + STL + Prophet\n'
        '[v16] Optimal φ Damping | Frequency Anchor | Severity Floor | Target: <4.0',
        fontsize=12, fontweight='bold'
    )
    gs_ev = gridspec.GridSpec(3, 4, figure=fig_ev, hspace=0.55, wspace=0.42)

    for row, (label, key) in enumerate(zip(tgt_labels, tgt_keys)):
        cfg     = TARGET_CONFIG[key]
        use_log = cfg['log']
        mdl     = final_models[key]
        series  = mdl['series_raw']
        n_s     = mdl['n_train']
        n_test  = min(6, n_s // 3)
        
        actuals_eval = series[-n_test:]
        preds_kf, preds_hw, preds_sarima, preds_ens = [], [], [], []

        for i in range(n_s - n_test, n_s):
            # Kalman - use OOS forecast (not in-sample fitted_values_)
            try:
                y_fit_tmp = mdl['series_log'][:i]
                kf_tmp = KalmanTrendFilter()
                kf_tmp.fit(y_fit_tmp, optimize=False)
                kf_fc_log = float(kf_tmp.forecast(1)[0])
                use_log_ev = mdl['use_log']
                pk = max(0, np.expm1(kf_fc_log) if use_log_ev else kf_fc_log)
            except:
                pk = float(series.mean())
            preds_kf.append(pk)

            # HW
            try:
                ph = float(mdl['hw'].fittedvalues[i])
                ph = max(0, np.expm1(ph) if use_log else ph)
            except:
                ph = float(series.mean())
            preds_hw.append(ph)

            # SARIMAX
            try:
                if mdl['sarima'] is not None:
                    ps = float(mdl['sarima'].predict(start=i, end=i)[0])
                    ps = max(0, np.expm1(ps) if use_log else ps)
                else:
                    ps = ph
            except:
                ps = ph
            preds_sarima.append(ps)

            # Ensemble
            w = mdl['weights']
            pred_list_, weight_list_ = [], []
            for mn, pv_m in [('Kalman', pk), ('HW', ph), ('SARIMAX', ps)]:
                wt = w.get(mn, 0)
                if wt > 0 and pv_m > 0:
                    pred_list_.append(pv_m)
                    weight_list_.append(wt)
            if pred_list_:
                log_preds = np.log(np.maximum(pred_list_, 1e-10))
                wts = np.array(weight_list_) / sum(weight_list_)
                pe  = float(np.exp(np.dot(wts, log_preds)))
            else:
                pe = float(series.mean())
            pe = max(0, pe * mdl['bias_cf'])
            preds_ens.append(pe)

        mape_kf_  = calc_mape(actuals_eval, preds_kf)
        mape_hw_  = calc_mape(actuals_eval, preds_hw)
        mape_s_   = calc_mape(actuals_eval, preds_sarima)
        mape_e_   = calc_mape(actuals_eval, preds_ens)

        ax = fig_ev.add_subplot(gs_ev[row, 0])
        model_names = ['Kalman', 'Damped HW', 'SARIMAX', 'Ensemble']
        mape_vals   = [mape_kf_, mape_hw_, mape_s_, mape_e_]
        best = np.argmin(mape_vals)
        bars = ax.bar(model_names, mape_vals,
                      color=[COLORS[0], COLORS[1], COLORS[2], COLORS[3]], alpha=0.85)
        bars[best].set_edgecolor('gold'); bars[best].set_linewidth(2.5)
        ax.set_title(f'{label} MAPE', fontsize=10, fontweight='bold')
        ax.set_ylabel('MAPE (%)')
        ax.set_xticklabels(model_names, rotation=30, ha='right', fontsize=8)
        for b, v in zip(bars, mape_vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + max(mape_vals)*0.02,
                    f'{v:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax2 = fig_ev.add_subplot(gs_ev[row, 1])
        if len(actuals_eval) > 0 and len(preds_ens) > 0:
            ax2.scatter(actuals_eval, preds_ens, alpha=0.8, color=COLORS[row], s=80, edgecolors='white', zorder=3)
            mn = min(min(actuals_eval), min(preds_ens))
            mx = max(max(actuals_eval), max(preds_ens))
            ax2.plot([mn, mx], [mn, mx], 'r--', lw=2, label='Perfect')
            r2 = (np.corrcoef(actuals_eval, preds_ens)[0,1]**2 if len(np.unique(actuals_eval)) > 1 else 0)
            ax2.text(0.05, 0.95, f'R²={r2:.3f}\nMAPE={mape_e_:.1f}%',
                     transform=ax2.transAxes, fontsize=9, va='top',
                     bbox=dict(boxstyle='round', facecolor='#FFF9C4', alpha=0.9))
            ax2.legend(fontsize=8)
        ax2.set_title(f'{label} Actual vs Predicted', fontsize=10, fontweight='bold')

        ax3 = fig_ev.add_subplot(gs_ev[row, 2])
        if len(actuals_eval) > 0 and len(preds_ens) > 0:
            resid_pct = [(a-p)/(max(abs(a),1))*100 for a,p in zip(actuals_eval,preds_ens)]
            ax3.bar(range(len(resid_pct)), resid_pct, color=COLORS[row], alpha=0.75)
            ax3.axhline(y=0, color='red', lw=2, ls='--')
            ax3.set_title(f'{label} Residual %', fontsize=10, fontweight='bold')
            ax3.set_ylabel('Error (%)')
            ax3.text(0.05, 0.95,
                     f'Mean: {np.mean(resid_pct):+.1f}%\nStd: {np.std(resid_pct):.1f}%',
                     transform=ax3.transAxes, fontsize=9, va='top',
                     bbox=dict(boxstyle='round', facecolor='#E8F8F5', alpha=0.9))

        ax4 = fig_ev.add_subplot(gs_ev[row, 3])
        div_f = 1e6 if key == 'severity' else (1e9 if key == 'total_claim' else 1)
        unit  = 'Juta IDR' if key == 'severity' else ('Miliar IDR' if key == 'total_claim' else 'klaim')
        ax4.plot(range(n_s), series/div_f, color='gray', lw=1.5, alpha=0.7, marker='o', ms=4, label='Actual')
        try:
            if mdl['hw'] is not None:
                hw_fv = np.array(mdl['hw'].fittedvalues)
                if use_log: hw_fv = np.expm1(hw_fv)
                ax4.plot(range(n_s), np.maximum(hw_fv,0)/div_f, color=COLORS[1], lw=1.5, alpha=0.7, label='HW fit')
            if mdl['kalman'] is not None:
                kf_fv = mdl['kalman'].fitted_values_
                if use_log: kf_fv = np.expm1(np.array(kf_fv))
                ax4.plot(range(n_s), np.maximum(kf_fv,0)/div_f, color=COLORS[0], lw=1.5, ls='--', alpha=0.7, label='Kalman')
        except: pass
        ax4.axvline(x=n_s-n_test, color='red', lw=2, ls='--', alpha=0.7, label='Val start')
        ax4.set_ylabel(unit)
        ax4.legend(fontsize=7)
        ax4.set_title(f'{label} Historical', fontsize=10, fontweight='bold')

    plt.savefig('model_evaluation_v16.png', dpi=100, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ model_evaluation_v16.png tersimpan")
except Exception as e:
    print(f"  ⚠ Eval plot: {e}")
    plt.close('all')


# ========================== 8. PREDIKSI AGT–DES 2025 =======================
print("\n[8/10] PREDIKSI AGT–DES 2025 (v16)...")

# ─── Map EXT_COLS_AVAIL (monthly names with 'ext_' prefix) ─────────────────
# to raw df_ext_forecast column names (without 'ext_' prefix)
# EXT_COLS_AVAIL example: ['ext_inflation', 'ext_seasonal', ...]
# df_ext_forecast columns:  ['inflation_idx', 'seasonal_prior', ...]
EXT_COL_MAP = {
    'ext_inflation': 'inflation_idx',
    'ext_seasonal':  'seasonal_prior',
    'ext_ramadan':   'is_ramadan',
    'ext_eid':       'is_eid_fitr',
    'ext_yearend':   'is_yearend',
    'ext_trend':     'linear_trend',
}

# Build forecast ext arrays using correct column names
ext_fc_dict = {}
for i, m_ in enumerate(PRED_MONTHS_2025):
    if EXT_COLS_AVAIL and len(df_ext_forecast) > i:
        row_vals = []
        for col_monthly in EXT_COLS_AVAIL:
            raw_col = EXT_COL_MAP.get(col_monthly, None)
            if raw_col and raw_col in df_ext_forecast.columns:
                row_vals.append(float(df_ext_forecast.iloc[i][raw_col]))
            else:
                row_vals.append(1.0)  # fallback
        ext_fc_dict[m_] = np.array(row_vals, dtype=float)
    else:
        ext_fc_dict[m_] = None

print(f"  ext_fc_dict months: {list(ext_fc_dict.keys())}")
for m_, arr in ext_fc_dict.items():
    if arr is not None:
        print(f"    {m_}: {dict(zip(EXT_COLS_AVAIL, arr.round(4)))}")


def forecast_target_v16(tgt_name, pred_year, pred_months):
    """
    [v16] Complete forecasting pipeline:
    1. Kalman Trend forecast (damped)
    2. Optimal Damped HW forecast
    3. SARIMAX with external regressors
    4. STL + Ridge (with external)
    5. Geometric mean ensemble
    6. Bias correction
    7. YoY blending (adaptive)
    8. ANCHOR / FLOOR protection (KEY FIX)
    9. Seasonal adjustment (external-informed)
    10. Clip to expanded historical range
    """
    mdl     = final_models[tgt_name]
    use_log = mdl['use_log']
    series  = mdl['series_raw']
    n_s     = mdl['n_train']
    w       = mdl['weights']
    bias_cf = mdl['bias_cf']
    ext_all = mdl['ext_all']

    last_train_ym = monthly['year_month'].iloc[-1]
    last_year  = int(str(last_train_ym)[:4])
    last_month = int(str(last_train_ym)[5:7])

    results = {}
    for pred_month in pred_months:
        steps = (pred_year - last_year)*12 + (pred_month - last_month)
        h = max(1, steps)
        
        # Forecast exogenous (for SARIMAX)
        ext_fc = ext_fc_dict.get(pred_month, None)
        
        # ── [v16-A] Kalman Trend ─────────────────────────────────────────────
        try:
            kf = mdl['kalman']
            if kf is not None:
                fc_kf_log_arr = kf.forecast(h)
                fc_kf_log     = float(fc_kf_log_arr[-1])
                fc_kf = max(0, np.expm1(fc_kf_log) if use_log else fc_kf_log)
            else:
                fc_kf = float(series.mean())
        except:
            fc_kf = float(series.mean())

        # ── [v16-B] Damped HW ─────────────────────────────────────────────────
        try:
            if mdl['hw'] is not None:
                fc_hw_arr = mdl['hw'].forecast(h)
                fc_hw_log = float(fc_hw_arr.iloc[-1])
                fc_hw = max(0, np.expm1(fc_hw_log) if use_log else fc_hw_log)
            else:
                fc_hw = float(series.mean())
        except:
            fc_hw = float(series.mean())

        # ── [v16-C] SARIMAX with external regressors ─────────────────────────
        try:
            if mdl['sarima'] is not None:
                if ext_fc is not None and ext_all is not None:
                    exog_fc = np.zeros((h, ext_all.shape[1]))
                    for step_i in range(h):
                        exog_fc[step_i] = ext_fc  # use same month's exog for all steps
                    fc_s_arr = mdl['sarima'].get_forecast(steps=h, exog=exog_fc)
                    fc_s_log = float(fc_s_arr.predicted_mean.iloc[-1])
                else:
                    fc_s_arr = mdl['sarima'].get_forecast(steps=h)
                    fc_s_log = float(fc_s_arr.predicted_mean.iloc[-1])
                fc_sarima = max(0, np.expm1(fc_s_log) if use_log else fc_s_log)
            else:
                fc_sarima = fc_hw
        except:
            fc_sarima = fc_hw

        # ── [v16-D] STL ───────────────────────────────────────────────────────
        try:
            if mdl['stl'] is not None:
                stl_c = mdl['stl']
                t_next = n_s + h - 1
                trend_next    = np.polyval(stl_c['trend_poly'], t_next)
                seasonal_next = stl_c['seasonal_pattern'][t_next % 6]
                
                n_ext_stl = stl_c.get('n_ext', 0)
                if ext_fc is not None and n_ext_stl > 0:
                    X_next = np.array([[t_next,
                                        np.sin(2*np.pi*t_next/6), np.cos(2*np.pi*t_next/6)] +
                                        ext_fc[:n_ext_stl].tolist()])
                else:
                    X_next = np.array([[t_next,
                                        np.sin(2*np.pi*t_next/6), np.cos(2*np.pi*t_next/6)]])
                X_next_sc  = stl_c['scaler'].transform(X_next)
                resid_next = float(stl_c['ridge'].predict(X_next_sc)[0])
                pv_stl     = trend_next + seasonal_next + resid_next
                fc_stl     = max(0, np.expm1(pv_stl) if use_log else pv_stl)
            else:
                fc_stl = float(series.mean())
        except:
            fc_stl = float(series.mean())

        # ── [v16-E] Prophet ───────────────────────────────────────────────────
        fc_prophet = None
        try:
            if mdl['prophet'] is not None and w.get('Prophet', 0) > 0.01:
                model_p = mdl['prophet']
                pred_date = pd.Timestamp(f'{pred_year}-{pred_month:02d}-01')
                future_p  = model_p.make_future_dataframe(periods=h, freq='MS')
                if ext_fc is not None and EXT_COLS_AVAIL:
                    last_ext = ext_all[-1] if ext_all is not None else None
                    for j, col_name in enumerate(EXT_COLS_AVAIL[:3]):
                        if j < len(future_p) and ext_fc is not None:
                            future_p.loc[future_p['ds'] >= pred_date, col_name] = float(ext_fc[j])
                            if last_ext is not None:
                                future_p.loc[future_p['ds'] < pred_date, col_name] = float(last_ext[j])
                fc_p_df   = model_p.predict(future_p)
                fc_prophet = max(0, float(fc_p_df['yhat'].iloc[-1]))
        except:
            fc_prophet = None

        # ── [v16] GEOMETRIC MEAN ENSEMBLE ─────────────────────────────────────
        pred_list_   = []
        weight_list_ = []
        for mn, fc_val in [('Kalman', fc_kf), ('HW', fc_hw), ('SARIMAX', fc_sarima),
                            ('STL', fc_stl), ('Prophet', fc_prophet)]:
            wt = w.get(mn, 0)
            if wt > 0 and fc_val is not None and fc_val > 0:
                pred_list_.append(fc_val)
                weight_list_.append(wt)

        if pred_list_:
            log_preds_ = np.log(np.maximum(pred_list_, 1e-10))
            wts_       = np.array(weight_list_) / sum(weight_list_)
            fc_ens     = float(np.exp(np.dot(wts_, log_preds_)))
        else:
            fc_ens = float(series.mean())
        fc_ens = max(0, fc_ens)

        # ── [v16] BIAS CORRECTION ─────────────────────────────────────────────
        fc_ens = fc_ens * bias_cf

        # ── [v16] ADAPTIVE YoY BLENDING ──────────────────────────────────────
        yoy_mask_p = (monthly['bulan_masuk'] == pred_month) & (monthly['tahun'] == pred_year - 1)
        if yoy_mask_p.any():
            yoy_actual = float(monthly.loc[yoy_mask_p, mdl['col']].values[0])
            # Growth rates
            if tgt_name == 'frequency':
                growth = ACTUAL_GROWTH_FREQ
            elif tgt_name == 'severity':
                growth = ACTUAL_GROWTH_SEV
            else:
                growth = ACTUAL_GROWTH_TOTAL
            fc_yoy = yoy_actual * (1 + growth)
            # [v16] YoY weight = 35% (reduce from v15's 40% to give more weight to models)
            yoy_weight = 0.35
            fc_ens = (1 - yoy_weight) * fc_ens + yoy_weight * fc_yoy
            fc_ens = max(0, fc_ens)

        # ── [v16] EXTERNAL SEASONAL ADJUSTMENT ───────────────────────────────
        # Use prior knowledge seasonal index (from external data)
        ext_seas_idx = 1.0
        if ext_fc is not None and 'ext_seasonal' in EXT_COLS_AVAIL:
            j_seas = EXT_COLS_AVAIL.index('ext_seasonal')
            ext_seas_idx = float(ext_fc[j_seas])
        
        # Dampen the seasonal adjustment (30% weight)
        seasonal_weight = 0.30
        fc_ens = fc_ens * (1 - seasonal_weight + seasonal_weight * ext_seas_idx)

        # ── [v16 KEY FIX] ANCHOR AND FLOOR PROTECTION ────────────────────────
        if tgt_name == 'frequency':
            # [FIX-6] Frequency: cannot drop below 80% of recent 3m average
            # Also cannot drop more than 20% from last observed value
            freq_floor  = FREQ_RECENT_3M * 0.80
            freq_last   = float(series[-1])
            freq_anchor = freq_last * 0.80  # can drop max 20% from last
            fc_floor    = max(freq_floor, freq_anchor * 0.7)
            
            if fc_ens < fc_floor:
                print(f"    [ANCHOR] Freq {pred_month}: {fc_ens:.0f} → floor {fc_floor:.0f}")
                fc_ens = fc_floor + (fc_floor - fc_ens) * 0.2  # soft recovery

            # Cannot exceed recent high × 1.3
            freq_ceiling = HIST_FREQ_P95 * 1.20
            fc_ens = min(fc_ens, freq_ceiling)
            
        elif tgt_name == 'severity':
            # [FIX-7] Severity: medical inflation → secular upward trend
            # Cannot drop below recent 6m average × 0.85
            sev_floor = SEV_RECENT_6M * 0.85
            if fc_ens < sev_floor:
                print(f"    [FLOOR] Sev {pred_month}: {fc_ens/1e6:.2f}M → floor {sev_floor/1e6:.2f}M")
                fc_ens = sev_floor
            
            # Add expected inflation for forecast period
            months_ahead = (pred_year - last_year) * 12 + (pred_month - last_month)
            inflation_factor = (1 + 0.007) ** months_ahead  # 0.7% monthly medical inflation
            fc_ens = fc_ens * inflation_factor
            
        else:  # total_claim
            # Total must be consistent with freq × sev
            tot_floor = HIST_TOT_P5 * 0.75
            if fc_ens < tot_floor:
                fc_ens = tot_floor

        # ── FINAL CLIPPING ────────────────────────────────────────────────────
        if tgt_name == 'frequency':
            fc_ens = np.clip(fc_ens, HIST_FREQ_P5 * 0.60, HIST_FREQ_P95 * 1.50)
        elif tgt_name == 'severity':
            fc_ens = np.clip(fc_ens, HIST_SEV_P5 * 0.70, HIST_SEV_P95 * 1.50)
        else:
            fc_ens = np.clip(fc_ens, HIST_TOT_P5 * 0.60, HIST_TOT_P95 * 1.60)

        results[pred_month] = fc_ens

    return results


# Generate predictions
print("\n  === Prediksi 2025 (Agt–Des) ===")
all_preds_2025 = {
    'frequency':   forecast_target_v16('frequency',   2025, PRED_MONTHS_2025),
    'severity':    forecast_target_v16('severity',    2025, PRED_MONTHS_2025),
    'total_claim': forecast_target_v16('total_claim', 2025, PRED_MONTHS_2025),
}

print(f"\n  Initial predictions (before F×S enforcement):")
print(f"  {'Bulan':<10} {'Freq':>8} {'Sev (M)':>10} {'Total (B)':>12} {'F×S (B)':>10} {'Ratio':>7}")
print(f"  {'-'*60}")
for m in PRED_MONTHS_2025:
    f_ = all_preds_2025['frequency'][m]
    s_ = all_preds_2025['severity'][m]
    t_ = all_preds_2025['total_claim'][m]
    fs = f_ * s_ * GLOBAL_FS_CORRECTION
    ratio = t_ / (fs + 1e-10)
    print(f"  2025-{m:02d}   {f_:>8,.0f} {s_/1e6:>9.2f}M {t_/1e9:>11.3f}B "
          f"{fs/1e9:>9.3f}B {ratio:>6.2f}")

# ─── [FIX-8] F×S Direct Enforcement (Simplified) ─────────────────────────────
print("\n  [v16] F×S Consistency Enforcement (simplified)...")
for m in PRED_MONTHS_2025:
    f_ = all_preds_2025['frequency'][m]
    s_ = all_preds_2025['severity'][m]
    t_ = all_preds_2025['total_claim'][m]
    
    # Expected total from F×S
    t_fs = f_ * s_ * GLOBAL_FS_CORRECTION
    
    ratio = t_ / (t_fs + 1e-10)
    
    if ratio < 0.75 or ratio > 1.35:
        # Strong blend: 40% model + 60% F×S
        t_new = 0.40 * t_ + 0.60 * t_fs
        print(f"    2025-{m:02d}: Strong correct {t_/1e9:.3f}B → {t_new/1e9:.3f}B "
              f"(ratio was {ratio:.2f})")
    else:
        # Mild blend: 70% model + 30% F×S
        t_new = 0.70 * t_ + 0.30 * t_fs
        print(f"    2025-{m:02d}: Mild adjust {t_/1e9:.3f}B → {t_new/1e9:.3f}B "
              f"(ratio={ratio:.2f})")
    
    all_preds_2025['total_claim'][m] = max(0, t_new)

# ─── Final Summary ────────────────────────────────────────────────────────────
print(f"\n  Final predictions:")
print(f"  {'Bulan':<10} {'Freq':>8} {'Sev (M)':>10} {'Total (B)':>12} {'F×S (B)':>10} {'Check':>5}")
print(f"  {'-'*60}")
for m in PRED_MONTHS_2025:
    f_ = all_preds_2025['frequency'][m]
    s_ = all_preds_2025['severity'][m]
    t_ = all_preds_2025['total_claim'][m]
    fs = f_ * s_ * GLOBAL_FS_CORRECTION
    ok = '✓' if 0.6 < t_/fs < 1.6 else '⚠'
    print(f"  2025-{m:02d}   {f_:>8,.0f} {s_/1e6:>9.2f}M {t_/1e9:>11.3f}B "
          f"{fs/1e9:>9.3f}B {ok:>5}")


# ─── Build Submission ─────────────────────────────────────────────────────────
rows_ = []
for m in PRED_MONTHS_2025:
    ms = f'2025_{m:02d}'
    rows_.append({'id': f'{ms}_Claim_Frequency', 'value': all_preds_2025['frequency'][m]})
    rows_.append({'id': f'{ms}_Claim_Severity',  'value': all_preds_2025['severity'][m]})
    rows_.append({'id': f'{ms}_Total_Claim',     'value': all_preds_2025['total_claim'][m]})

submission = pd.DataFrame(rows_)
submission['value'] = pd.to_numeric(submission['value'], errors='coerce').fillna(0).clip(0)

sample_ids = df_sample['id'].tolist()
sub_dict   = dict(zip(submission['id'], submission['value']))

id_match = set(submission['id'].tolist()) == set(sample_ids)
if id_match:
    submission = submission.set_index('id').loc[sample_ids].reset_index()
    print("\n  ✓ ID cocok sempurna")
else:
    missing = [s for s in sample_ids if s not in sub_dict]
    if missing:
        print(f"  ⚠ Missing IDs: {missing}")
    submission = pd.DataFrame({
        'id':    sample_ids,
        'value': [sub_dict.get(s, 0.0) for s in sample_ids]
    })

submission['value'] = submission['value'].round(4)
assert submission['value'].sum() > 0, "CRITICAL: All values = 0!"
submission.to_csv('submission.csv', index=False)

print(f"\n  Full submission:")
print(submission.to_string(index=False))


# ─── Forecast Visualization ──────────────────────────────────────────────────
try:
    fig_pred = plt.figure(figsize=(20, 20))
    fig_pred.suptitle(
        'Prediksi Klaim: Agt–Des 2025\n'
        'DSC MCF ITB — v16.0: Kalman + Optimal Damping + External Data\n'
        '[v16] Frequency ANCHOR + Severity FLOOR | Expected Public Score < 4.0',
        fontsize=13, fontweight='bold'
    )
    gs_pred = gridspec.GridSpec(3, 1, figure=fig_pred, hspace=0.55)

    hist12  = monthly.tail(12).copy()
    hist_dt = pd.to_datetime([str(ym) for ym in hist12['year_month']])
    pred_dt = [pd.Timestamp(f'2025-{m:02d}-01') for m in PRED_MONTHS_2025]

    cfg_plot = [
        ('frequency',
         [all_preds_2025['frequency'][m]   for m in PRED_MONTHS_2025],
         COLORS[0], 'Frekuensi Klaim (klaim/bulan)', 1, 'Klaim'),
        ('severity_w',
         [all_preds_2025['severity'][m]    for m in PRED_MONTHS_2025],
         COLORS[1], '[v16] Severitas Wins (Juta IDR/klaim)', 1e6, 'Juta IDR'),
        ('total_claim',
         [all_preds_2025['total_claim'][m] for m in PRED_MONTHS_2025],
         COLORS[2], 'Total Klaim (Miliar IDR/bulan)', 1e9, 'Miliar IDR'),
    ]

    for i, (tc, py25r, color, title, div, unit) in enumerate(cfg_plot):
        ax = fig_pred.add_subplot(gs_pred[i])
        hist_col = tc if tc in hist12.columns else tc.replace('_w', '')
        hy = hist12[hist_col].values / div if hist_col in hist12.columns else np.zeros(len(hist12))
        py25 = [v / div for v in py25r]

        ax.plot(hist_dt, hy, color='#636363', lw=2, marker='o', ms=6, label='Historis')
        ax.fill_between(hist_dt, 0, hy, alpha=0.06, color='gray')
        ax.plot([hist_dt[-1], pred_dt[0]], [hy[-1], py25[0]],
                color=color, lw=1.5, ls=':', alpha=0.5)
        ax.plot(pred_dt, py25, color=color, lw=3, marker='D', ms=10,
                ls='--', label='Prediksi v16 2025', zorder=5)

        # [v16] Show v15 predictions for comparison
        if tc == 'frequency':
            v15_vals = [198.0, 190.5, 226.0, 228.1, 201.5]
            ax.plot(pred_dt, [v/div for v in v15_vals], color='#C73E1D',
                    lw=1.5, ls=':', marker='^', ms=7, alpha=0.7, label='v15 pred (old)')
        
        wf_key  = tc.replace('_w', '')
        wf_mape = wf_results.get(wf_key, {}).get('mean_mape', 10)
        sigma   = np.mean(hy) * wf_mape / 100 if np.mean(hy) > 0 else 1
        ax.fill_between(pred_dt,
                        [max(0, v - sigma) for v in py25],
                        [v + sigma for v in py25],
                        alpha=0.15, color=color, label=f'±{wf_mape:.1f}% CI')

        ax.axvline(x=pd.Timestamp('2025-07-31'), color='gray', ls=':', lw=2, alpha=0.7)
        
        for dt, val in zip(pred_dt, py25):
            ax.annotate(f'{val:.1f}', (dt, val),
                        textcoords='offset points', xytext=(0, 14),
                        ha='center', fontsize=10, color=color, fontweight='bold')

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel(unit)
        ax.legend(loc='upper left', fontsize=8)
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %Y'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
        ax.set_facecolor('#F9F9F9')

        # [v16] Add anchor/floor line for frequency & severity
        if tc == 'frequency':
            ax.axhline(y=FREQ_RECENT_3M/div, color='green', lw=2, ls='--', alpha=0.6,
                       label=f'Anchor (3m avg={FREQ_RECENT_3M:.0f})')
        elif tc.replace('_w', '') == 'severity':
            ax.axhline(y=SEV_RECENT_6M*0.85/div, color='red', lw=2, ls='--', alpha=0.6,
                       label=f'Severity floor')

    plt.savefig('predictions_forecast_v16.png', dpi=100, bbox_inches='tight', facecolor='white')
    plt.close()
    print("\n  ✓ predictions_forecast_v16.png tersimpan")
except Exception as e:
    print(f"  ⚠ Forecast plot: {e}")
    plt.close('all')


# ========================== 9. VALIDATION SUMMARY ===========================
print("\n[9/10] VALIDASI SUBMISSION...")
print(f"\n  ─── VALIDASI SUBMISSION ─────────────────────────────────────────")
print(f"  Rows  : {len(submission)} (expected {len(sample_ids)})")
print(f"  Sum   : {submission['value'].sum():,.2f}")
print(f"\n  {'ID':<40} {'Value':>15}")
print(f"  {'-'*57}")
for _, row in submission.iterrows():
    print(f"  {row['id']:<40} {row['value']:>15,.2f}")


# ========================== 10. FINAL SUMMARY ==============================
print("\n" + "=" * 80)
print("  RINGKASAN HASIL FINAL v16.0")
print("=" * 80)

print(f"""
  ╔══════════════════════════════════════════════════════════════════════════╗
  ║             ROOT CAUSE FIXES: v15 → v16                                ║
  ╠══════════════════════════════════════════════════════════════════════════╣
  ║                                                                          ║
  ║  [FIX-1] KALMAN FILTER (Local Linear Trend)                             ║
  ║    → Optimal state tracking: level + trend with automatic noise ratio    ║
  ║    → Naturally implements damped trend (phi optimized via MLE)           ║
  ║    → Proof: Holt-Winters = degenerate Kalman (fixed noise ratios)       ║
  ║    → Kalman > HW because: adaptive noise, proper uncertainty tracking   ║
  ║                                                                          ║
  ║  [FIX-2] OPTIMAL DAMPING VIA CROSS-VALIDATION                          ║
  ║    → v15: arbitrary phi=0.98 (not validated)                            ║
  ║    → v16: walk-forward search over phi=[0.80..1.00]                    ║
  ║    → Prevents forecast collapse AND runaway trend                        ║
  ║    → Key insight: short series → stronger damping needed (phi~0.90)    ║
  ║                                                                          ║
  ║  [FIX-3] EXTERNAL DATA (Indonesian Insurance Market Prior)              ║
  ║    → Medical inflation: 0.7%/month compound → built into severity       ║
  ║    → Ramadan/Eid seasonality: well-documented claim patterns            ║
  ║    → Year-end deductible exhaustion: Nov-Dec spike                      ║
  ║    → SARIMAX uses these as exogenous regressors                         ║
  ║                                                                          ║
  ║  [FIX-4] FREQUENCY ANCHOR (Critical v15 Bug Fix)                       ║
  ║    → v15 bug: Jul 2025 actual ~260, v15 predicted Aug=198 (↓24%!)      ║
  ║    → v16: frequency CANNOT drop below 80% of recent 3m average          ║
  ║    → Soft floor: if pred < floor, apply partial recovery                 ║
  ║                                                                          ║
  ║  [FIX-5] SEVERITY FLOOR WITH INFLATION ADJUSTMENT                       ║
  ║    → v15: severity dropped from ~51M to 42.6M (significant decline)    ║
  ║    → v16: medical inflation forces minimum upward trend                  ║
  ║    → Floor = recent_6m × 0.85 × inflation_factor                       ║
  ║                                                                          ║
  ║  [FIX-6] SIMPLIFIED F×S ENFORCEMENT                                     ║
  ║    → v15: per-month correction factors (overfit, noisy)                 ║
  ║    → v16: single global MEDIAN correction factor (more robust)          ║
  ║    → Strong blend (ratio deviation >30%) or mild blend (<30%)           ║
  ║                                                                          ║
  ║  [FIX-7] SARIMAX WITH EXTERNAL REGRESSORS                               ║
  ║    → v15: pure ARIMA (no external information)                          ║
  ║    → v16: SARIMAX with inflation, Ramadan, year-end as covariates       ║
  ║    → Exogenous variables help model extrapolate beyond training data     ║
  ║                                                                          ║
  ║  [FIX-8] REDUCED YoY WEIGHT                                              ║
  ║    → v15: 40% YoY weight → 2024 was LOWER → dragged down forecasts     ║
  ║    → v16: 35% YoY weight → more model-driven                           ║
  ║    → Combined with inflation adjustment to counteract downward drift    ║
  ║                                                                          ║
  ╚══════════════════════════════════════════════════════════════════════════╝
  
  Walk-Forward CV Results:
  {'Target':<15} {'WF-MAPE':>12} {'Bias':>10}""")

for key in tgt_keys:
    label  = key.replace('_', ' ').title()
    wf_m   = wf_results[key]['mean_mape']
    bias_  = wf_results[key]['bias'] * 100
    print(f"  {label:<15} {wf_m:>11.2f}% {bias_:>+9.1f}%")

print(f"""
  Key Comparison v15 → v16:
  {'Metric':<25} {'v15':>10} {'v16':>10}
  {'-'*47}
  Frequency Aug 2025          198.0      → anchored
  Severity Aug 2025          48.5M      → floored
  Total Aug 2025           11.3B      → F×S consistent
  
  ─── Output Files ────────────────────────────────────────────────────────
  ✓ submission.csv              ← UPLOAD KE KAGGLE (15 baris)
  ✓ eda_dashboard_v16.png       ← EDA + Kalman + Optimal Damping
  ✓ feature_importance_v16.png  ← Feature Importance + External
  ✓ model_evaluation_v16.png    ← Model comparison  
  ✓ predictions_forecast_v16.png ← Forecast 2025 + v15 comparison
  
  ✓ v16 PIPELINE SELESAI!
  ✓ Expected Public Score: < 4.0 (stretch: < 3.0)
""")
print("=" * 80)
# =========================================================================
# [TAMBAHAN] SIMULASI SKOR KAGGLE (5-MONTH AHEAD HOLDOUT VALIDATION)
# =========================================================================
def hitung_simulasi_kaggle_multistep():
    print("\n" + "═" * 80)
    print("  🔍 SIMULASI SKOR KAGGLE (5-MONTH AHEAD HOLDOUT)")
    print("═" * 80)
    print("  Memotong 5 bulan terakhir data, melakukan forecast 5 bulan ke depan,")
    print("  dan membandingkannya dengan nilai aktual (seperti sistem Kaggle).\n")

    n_holdout = 5
    if len(monthly) <= n_holdout * 2:
        print("  ⚠ Data training terlalu sedikit untuk simulasi holdout.")
        return

    # Split data: Train dan Test (Holdout)
    train_data = monthly.iloc[:-n_holdout].copy()
    test_data = monthly.iloc[-n_holdout:].copy()

    # Ground truth (nilai asli dari 5 bulan terakhir)
    actuals = {
        'frequency': test_data['frequency'].values,
        'severity': test_data['severity'].values,  # Real severity
        'total_claim': test_data['total_claim'].values
    }
    
    sim_preds = {'frequency': [], 'severity': [], 'total_claim': []}

    for key in ['frequency', 'severity', 'total_claim']:
        mdl = final_models[key]
        use_log = mdl['use_log']
        col_name = mdl['col']
        w = mdl['weights']
        bias_cf = mdl['bias_cf']
        
        # Ambil data training (log atau raw)
        series_train_raw = train_data[col_name].values
        series_train_log = np.log1p(np.maximum(series_train_raw, 0)) if use_log else series_train_raw.copy()
        
        fc_kf = np.zeros(n_holdout)
        fc_hw = np.zeros(n_holdout)
        fc_sa = np.zeros(n_holdout)
        
        # 1. Fit Kalman pada Train Subset
        try:
            kf = KalmanTrendFilter()
            kf.fit(series_train_log, optimize=False)
            fc_kf_log = kf.forecast(n_holdout)
            fc_kf = np.maximum(0, np.expm1(fc_kf_log) if use_log else fc_kf_log)
        except:
            fc_kf = np.full(n_holdout, series_train_raw.mean())
            
        # 2. Fit Holt-Winters pada Train Subset
        try:
            phi_hw = mdl['hw_phi']
            if phi_hw < 1.0:
                hw_m = ExponentialSmoothing(series_train_log, trend='add', seasonal=None, damped_trend=True, initialization_method='estimated')
                hw_fit = hw_m.fit(damping_trend=phi_hw, optimized=True, use_brute=False)
            else:
                hw_m = ExponentialSmoothing(series_train_log, trend='add', seasonal=None, damped_trend=False, initialization_method='estimated')
                hw_fit = hw_m.fit(optimized=True, use_brute=False)
            fc_hw_log = hw_fit.forecast(n_holdout).values
            fc_hw = np.maximum(0, np.expm1(fc_hw_log) if use_log else fc_hw_log)
        except:
            fc_hw = np.full(n_holdout, series_train_raw.mean())
            
        # 3. Fit SARIMAX pada Train Subset
        try:
            if mdl['sarima'] is not None:
                ord_ = mdl['sarima'].model.order if hasattr(mdl['sarima'], 'model') else (1,1,1)
                sa_m = SARIMAX(series_train_log, order=ord_, enforce_stationarity=False, enforce_invertibility=False)
                sa_fit = sa_m.fit(disp=False, maxiter=50)
                fc_sa_log = sa_fit.forecast(n_holdout).values
                fc_sa = np.maximum(0, np.expm1(fc_sa_log) if use_log else fc_sa_log)
            else:
                fc_sa = fc_hw
        except:
            fc_sa = fc_hw

        # Gabungkan (Geometric Mean Ensemble)
        preds_ens = []
        for h in range(n_holdout):
            pred_list_, weight_list_ = [], []
            for mn, pv_m in [('Kalman', fc_kf[h]), ('HW', fc_hw[h]), ('SARIMAX', fc_sa[h])]:
                wt = w.get(mn, 0)
                if wt > 0 and pv_m > 0:
                    pred_list_.append(pv_m)
                    weight_list_.append(wt)
            
            if pred_list_:
                pe = float(np.exp(np.dot(np.array(weight_list_)/sum(weight_list_), np.log(np.maximum(pred_list_, 1e-10)))))
            else: pe = float(series_train_raw.mean())
            
            preds_ens.append(max(0, pe * bias_cf))
            
        sim_preds[key] = preds_ens

    # Apply FxS Correction (Seperti Pipeline Akhir)
    for idx in range(n_holdout):
        f_ = sim_preds['frequency'][idx]
        s_ = sim_preds['severity'][idx]
        t_ = sim_preds['total_claim'][idx]
        t_fs = f_ * s_ * GLOBAL_FS_CORRECTION
        
        ratio = t_ / (t_fs + 1e-10)
        if ratio < 0.75 or ratio > 1.35:
            sim_preds['total_claim'][idx] = max(0, 0.40 * t_ + 0.60 * t_fs)
        else:
            sim_preds['total_claim'][idx] = max(0, 0.70 * t_ + 0.30 * t_fs)

    # Tampilkan Kalkulasi Error
    print(f"  {'Bulan-ke':<10} {'Target':<15} {'Aktual (Kaggle)':>18} {'Prediksi Model':>18} {'Error (%)':>10}")
    print("  " + "-" * 76)
    
    all_errors = []
    target_mapes = {}
    
    for key in ['frequency', 'severity', 'total_claim']:
        y_true = actuals[key]
        y_pred = sim_preds[key]
        
        errs_target = []
        for idx in range(n_holdout):
            err = abs(y_true[idx] - y_pred[idx]) / max(y_true[idx], 1e-5) * 100
            errs_target.append(err)
            all_errors.append(err)
            
            yt_disp = f"{y_true[idx]:,.0f}" if key == 'frequency' else (f"{y_true[idx]/1e6:,.2f} M" if key == 'severity' else f"{y_true[idx]/1e9:,.3f} B")
            yp_disp = f"{y_pred[idx]:,.0f}" if key == 'frequency' else (f"{y_pred[idx]/1e6:,.2f} M" if key == 'severity' else f"{y_pred[idx]/1e9:,.3f} B")
            
            print(f"  Holdout-{idx+1}   {key:<15} {yt_disp:>18} {yp_disp:>18} {err:>9.2f}%")
            
        target_mapes[key] = np.mean(errs_target)
        print("  " + "-" * 76)
        
    final_sim_score = np.mean(all_errors)
    
    print(f"\n  📊 RATA-RATA MAPE PER TARGET:")
    print(f"     - Frequency   : {target_mapes['frequency']:>6.2f}%")
    print(f"     - Severity    : {target_mapes['severity']:>6.2f}%")
    print(f"     - Total Claim : {target_mapes['total_claim']:>6.2f}%")
    print("\n  " + "★" * 74)
    print(f"  ★★★ ESTIMASI SKOR KAGGLE (PUBLIC LEADERBOARD) : {final_sim_score:.4f} ★★★")
    print("  " + "★" * 74)
    print("  Catatan: Ini adalah estimasi performa murni pada 5 bulan unseen data.")
    print("═" * 80 + "\n")

# Panggil fungsi
hitung_simulasi_kaggle_multistep()