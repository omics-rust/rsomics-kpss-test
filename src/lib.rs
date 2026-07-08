//! KPSS (Kwiatkowski-Phillips-Schmidt-Shin) stationarity test.
//!
//! Null hypothesis: the series is stationary (level or trend). Opposite of ADF.
//!
//! Algorithm (Kwiatkowski et al. 1992, Table 1):
//! 1. Detrend: regression="c" → subtract mean; regression="ct" → OLS on [1, t].
//! 2. Compute cumulative residual sums: S_t = Σ_{i=1}^{t} e_i.
//! 3. HAC variance: σ² = (Σ e_i² + 2 Σ_{j=1}^{l} w_j Σ e_t e_{t-j}) / n
//!    where w_j = 1 - j/(l+1) (Bartlett kernel).
//! 4. Test statistic: η = (Σ S_t²) / (n² · σ²).
//! 5. P-value: linear interpolation in Kwiatkowski et al. (1992) Table 1.

use rsomics_common::{Result, RsomicsError};

/// Regression type for the KPSS detrending step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Regression {
    /// Level stationarity: detrend by subtracting the mean.
    C,
    /// Trend stationarity: detrend by OLS on [intercept, linear trend].
    Ct,
}

impl Regression {
    /// Parse "c" or "ct" (case-insensitive).
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "c" => Some(Self::C),
            "ct" => Some(Self::Ct),
            _ => None,
        }
    }
}

/// Lag selection method for the HAC estimator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NLags {
    /// Hobijn et al. (1998) data-dependent selection.
    Auto,
    /// Schwert (1989): ceil(12 * (n/100)^0.25), clamped to n-1.
    Legacy,
    /// User-specified fixed lag count.
    Fixed(usize),
}

impl NLags {
    /// Parse "auto", "legacy", or an integer string.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "legacy" => Some(Self::Legacy),
            _ => s.parse::<usize>().ok().map(Self::Fixed),
        }
    }
}

/// Result of the KPSS test.
#[derive(Debug, Clone)]
pub struct KpssResult {
    /// The KPSS test statistic (η).
    pub kpss_stat: f64,
    /// Interpolated p-value from Kwiatkowski et al. (1992) Table 1.
    pub pvalue: f64,
    /// Number of lags used for the HAC estimator.
    pub lags: usize,
    /// Critical values at 10%, 5%, 2.5%, 1%.
    pub crit_10pct: f64,
    pub crit_5pct: f64,
    pub crit_2_5pct: f64,
    pub crit_1pct: f64,
}

/// Perform the KPSS stationarity test.
///
/// Matches `statsmodels.tsa.stattools.kpss` output exactly on valid input, and
/// fails loud where statsmodels raises: non-finite values, fewer than three
/// observations, a fixed lag count `>= nobs`, or a zero residual variance
/// (constant / perfectly detrended series, where the statistic is undefined).
pub fn kpss(x: &[f64], regression: Regression, nlags: &NLags) -> Result<KpssResult> {
    let nobs = x.len();
    if nobs < 3 {
        return Err(RsomicsError::InvalidInput(format!(
            "kpss: series must have at least 3 observations, got {nobs}"
        )));
    }
    if let Some(&bad) = x.iter().find(|v| !v.is_finite()) {
        return Err(RsomicsError::InvalidInput(format!(
            "kpss: input contains a non-finite value ({bad})"
        )));
    }

    // Step 1: detrend residuals.
    let resids = detrend(x, regression);

    // Critical value tables from Kwiatkowski et al. (1992) Table 1, row "∞ obs".
    // p-values: [0.10, 0.05, 0.025, 0.01].
    let (crit, crit_arr) = match regression {
        Regression::C => (
            [0.347_f64, 0.463, 0.574, 0.739],
            [0.347_f64, 0.463, 0.574, 0.739],
        ),
        Regression::Ct => (
            [0.119_f64, 0.146, 0.176, 0.216],
            [0.119_f64, 0.146, 0.176, 0.216],
        ),
    };
    let pvals = [0.10_f64, 0.05, 0.025, 0.01];

    // Step 2: determine lag count.
    let lags = match nlags {
        NLags::Auto => {
            let l = kpss_autolag(&resids, nobs);
            l.min(nobs - 1)
        }
        NLags::Legacy => {
            let l = (12.0 * (nobs as f64 / 100.0).powf(0.25)).ceil() as usize;
            l.min(nobs - 1)
        }
        NLags::Fixed(l) => {
            if *l >= nobs {
                return Err(RsomicsError::InvalidInput(format!(
                    "nlags ({l}) must be < number of observations ({nobs})"
                )));
            }
            *l
        }
    };

    // Step 3: cumulative sum statistic η (eq. 11, p. 165).
    let cumsum: Vec<f64> = resids
        .iter()
        .scan(0.0_f64, |acc, &r| {
            *acc += r;
            Some(*acc)
        })
        .collect();
    let eta = cumsum.iter().map(|s| s * s).sum::<f64>() / (nobs as f64 * nobs as f64);

    // Step 4: HAC variance σ² (eq. 10, p. 164).
    let sigma_sq = sigma_est_kpss(&resids, nobs, lags);

    // A constant or perfectly detrended series has zero residual variance, so
    // η/σ² is 0/0 = NaN and the statistic is undefined. statsmodels raises here
    // (regression="c") or returns lstsq rounding noise (regression="ct"); we
    // fail loud rather than emit a NaN dressed as a success.
    if sigma_sq == 0.0 {
        return Err(RsomicsError::InvalidInput(
            "kpss: residual variance is zero (series is constant or perfectly \
             detrended); the test statistic is undefined"
                .into(),
        ));
    }

    let kpss_stat = eta / sigma_sq;

    // Step 5: p-value by linear interpolation in crit table.
    // np.interp(x, xp, fp): xp = crit values (ascending), fp = pvals.
    // statsmodels uses crit as xp and pvals as fp.
    let pvalue = interp(kpss_stat, &crit_arr, &pvals);

    Ok(KpssResult {
        kpss_stat,
        pvalue,
        lags,
        crit_10pct: crit[0],
        crit_5pct: crit[1],
        crit_2_5pct: crit[2],
        crit_1pct: crit[3],
    })
}

/// Detrend the series according to the regression type.
///
/// "c": subtract mean. "ct": OLS on [1, t] (t = 1..nobs), return residuals.
fn detrend(x: &[f64], regression: Regression) -> Vec<f64> {
    match regression {
        Regression::C => {
            let mean = x.iter().sum::<f64>() / x.len() as f64;
            x.iter().map(|v| v - mean).collect()
        }
        Regression::Ct => {
            // OLS: X = [[1, 1], [1, 2], ..., [1, n]], Y = x.
            // t = 1..=nobs (1-based, matching numpy arange(1, nobs+1)).
            let n = x.len() as f64;
            let nobs = x.len();
            // Precompute sums for the 2×2 normal equations.
            // X'X = [[n, Σt], [Σt, Σt²]]   where t=1..nobs
            // X'Y = [Σy, Σty]
            let sum_t = n * (n + 1.0) / 2.0; // Σ_{t=1}^{n} t
            let sum_t2 = n * (n + 1.0) * (2.0 * n + 1.0) / 6.0; // Σ t²
            let sum_y: f64 = x.iter().sum();
            let sum_ty: f64 = x.iter().enumerate().map(|(i, &y)| (i + 1) as f64 * y).sum();
            // Solve 2×2 system: [n, sum_t; sum_t, sum_t2] * [b0, b1]' = [sum_y, sum_ty]'
            let det = n * sum_t2 - sum_t * sum_t;
            let b0 = (sum_t2 * sum_y - sum_t * sum_ty) / det;
            let b1 = (n * sum_ty - sum_t * sum_y) / det;
            (0..nobs)
                .map(|i| x[i] - (b0 + b1 * (i + 1) as f64))
                .collect()
        }
    }
}

/// HAC variance estimator (Newey-West / Bartlett, eq. 10 p. 164).
///
/// σ² = (Σ e_i² + 2 Σ_{j=1}^{l} (1-j/(l+1)) Σ e_t e_{t-j}) / n
fn sigma_est_kpss(resids: &[f64], nobs: usize, lags: usize) -> f64 {
    let mut s: f64 = resids.iter().map(|e| e * e).sum();
    for j in 1..=lags {
        // dot product of resids[j..] with resids[..nobs-j]
        let dot: f64 = resids[j..]
            .iter()
            .zip(&resids[..nobs - j])
            .map(|(a, b)| a * b)
            .sum();
        let w = 1.0 - j as f64 / (lags as f64 + 1.0);
        s += 2.0 * dot * w;
    }
    s / nobs as f64
}

/// Autolag selection (Hobijn et al. 1998).
fn kpss_autolag(resids: &[f64], nobs: usize) -> usize {
    let covlags = (nobs as f64).powf(2.0 / 9.0) as usize;
    let mut s0: f64 = resids.iter().map(|e| e * e).sum::<f64>() / nobs as f64;
    let mut s1 = 0.0_f64;
    for i in 1..=covlags {
        let dot: f64 = resids[i..]
            .iter()
            .zip(&resids[..nobs - i])
            .map(|(a, b)| a * b)
            .sum::<f64>();
        let rp = dot / (nobs as f64 / 2.0);
        s0 += rp;
        s1 += i as f64 * rp;
    }
    let s_hat = s1 / s0;
    let gamma_hat = 1.1447 * (s_hat * s_hat).powf(1.0 / 3.0);
    (gamma_hat * (nobs as f64).powf(1.0 / 3.0)) as usize
}

/// Linear interpolation matching `numpy.interp(x, xp, fp)`.
///
/// If x < xp[0] → fp[0]; if x > xp[-1] → fp[-1]; otherwise linear.
fn interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    if x <= xp[0] {
        return fp[0];
    }
    if x >= xp[xp.len() - 1] {
        return fp[fp.len() - 1];
    }
    // Find interval [xp[i], xp[i+1]] containing x.
    let mut lo = 0;
    while lo + 1 < xp.len() - 1 && xp[lo + 1] <= x {
        lo += 1;
    }
    let t = (x - xp[lo]) / (xp[lo + 1] - xp[lo]);
    fp[lo] + t * (fp[lo + 1] - fp[lo])
}
