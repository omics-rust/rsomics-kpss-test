//! Value-exact compatibility against `statsmodels.tsa.stattools.kpss` 0.14.6.
//!
//! All goldens computed with Python oracle (numpy seed 42, statsmodels 0.14.6)
//! and frozen below. No Python or subprocess at test time.

#![allow(clippy::excessive_precision)]

use rsomics_kpss_test::{NLags, Regression, kpss};

include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/series_data.rs"));

fn assert_close(got: f64, want: f64, tol: f64, label: &str) {
    let e = if want.abs() < 1e-300 {
        (got - want).abs()
    } else {
        ((got - want) / want).abs()
    };
    assert!(
        e < tol,
        "{label}: got {got:.17e}, want {want:.17e}, rel_err {e:.3e}"
    );
}

// kpss(rw, c, auto): stat=0.7001396636588891 pvalue=0.013532757849191895 lags=9
#[test]
fn rw_c_auto() {
    let r = kpss(&RW, Regression::C, &NLags::Auto);
    assert_close(
        r.kpss_stat,
        0.7001396636588891,
        1e-10,
        "rw_c_auto kpss_stat",
    );
    assert_close(r.pvalue, 0.013532757849191895, 1e-10, "rw_c_auto pvalue");
    assert_eq!(r.lags, 9, "rw_c_auto lags");
    assert_close(r.crit_10pct, 0.347, 1e-10, "rw_c_auto crit_10pct");
    assert_close(r.crit_5pct, 0.463, 1e-10, "rw_c_auto crit_5pct");
    assert_close(r.crit_2_5pct, 0.574, 1e-10, "rw_c_auto crit_2_5pct");
    assert_close(r.crit_1pct, 0.739, 1e-10, "rw_c_auto crit_1pct");
}

// kpss(wn, c, auto): stat=0.0743681910503493 pvalue=0.1 lags=3
#[test]
fn wn_c_auto() {
    let r = kpss(&WN, Regression::C, &NLags::Auto);
    assert_close(
        r.kpss_stat,
        0.0743681910503493,
        1e-10,
        "wn_c_auto kpss_stat",
    );
    assert_close(r.pvalue, 0.1, 1e-10, "wn_c_auto pvalue");
    assert_eq!(r.lags, 3, "wn_c_auto lags");
}

// kpss(rw, ct, auto): stat=0.3664054315050512 pvalue=0.01 lags=9
#[test]
fn rw_ct_auto() {
    let r = kpss(&RW, Regression::Ct, &NLags::Auto);
    assert_close(
        r.kpss_stat,
        0.3664054315050512,
        1e-10,
        "rw_ct_auto kpss_stat",
    );
    assert_close(r.pvalue, 0.01, 1e-10, "rw_ct_auto pvalue");
    assert_eq!(r.lags, 9, "rw_ct_auto lags");
    assert_close(r.crit_10pct, 0.119, 1e-10, "rw_ct_auto crit_10pct");
    assert_close(r.crit_5pct, 0.146, 1e-10, "rw_ct_auto crit_5pct");
    assert_close(r.crit_2_5pct, 0.176, 1e-10, "rw_ct_auto crit_2_5pct");
    assert_close(r.crit_1pct, 0.216, 1e-10, "rw_ct_auto crit_1pct");
}

// kpss(ar1, c, auto): stat=0.0775610775880112 pvalue=0.1 lags=6
#[test]
fn ar1_c_auto() {
    let r = kpss(&AR1, Regression::C, &NLags::Auto);
    assert_close(
        r.kpss_stat,
        0.0775610775880112,
        1e-10,
        "ar1_c_auto kpss_stat",
    );
    assert_close(r.pvalue, 0.1, 1e-10, "ar1_c_auto pvalue");
    assert_eq!(r.lags, 6, "ar1_c_auto lags");
}

// kpss(rw, c, legacy): stat=0.480572984722357 pvalue=0.04604212055802771 lags=15
#[test]
fn rw_c_legacy() {
    let r = kpss(&RW, Regression::C, &NLags::Legacy);
    assert_close(
        r.kpss_stat,
        0.480572984722357,
        1e-10,
        "rw_c_legacy kpss_stat",
    );
    assert_close(r.pvalue, 0.04604212055802771, 1e-10, "rw_c_legacy pvalue");
    assert_eq!(r.lags, 15, "rw_c_legacy lags");
}

// kpss(rw, c, nlags=5): stat=1.1054946417223224 pvalue=0.01 lags=5
#[test]
fn rw_c_fixed5() {
    let r = kpss(&RW, Regression::C, &NLags::Fixed(5));
    assert_close(
        r.kpss_stat,
        1.1054946417223224,
        1e-10,
        "rw_c_fixed5 kpss_stat",
    );
    assert_close(r.pvalue, 0.01, 1e-10, "rw_c_fixed5 pvalue");
    assert_eq!(r.lags, 5, "rw_c_fixed5 lags");
}

// kpss(wn, ct, auto): stat=0.06212221588515339 pvalue=0.1 lags=3
#[test]
fn wn_ct_auto() {
    let r = kpss(&WN, Regression::Ct, &NLags::Auto);
    assert_close(
        r.kpss_stat,
        0.06212221588515339,
        1e-10,
        "wn_ct_auto kpss_stat",
    );
    assert_close(r.pvalue, 0.1, 1e-10, "wn_ct_auto pvalue");
    assert_eq!(r.lags, 3, "wn_ct_auto lags");
}
