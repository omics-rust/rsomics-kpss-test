use criterion::{Criterion, criterion_group, criterion_main};
use rsomics_kpss_test::{NLags, Regression, kpss};

fn randn_lcg(n: usize, seed: u64) -> Vec<f64> {
    // Simple LCG + Box-Muller for reproducible bench data
    let mut s = seed;
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u1 = (s >> 32) as f64 / u32::MAX as f64;
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u2 = (s >> 32) as f64 / u32::MAX as f64;
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        out.push(z);
    }
    out
}

fn bench_kpss(c: &mut Criterion) {
    let noise: Vec<f64> = randn_lcg(2000, 42);
    let rw: Vec<f64> = noise
        .iter()
        .scan(0.0_f64, |acc, &x| {
            *acc += x;
            Some(*acc)
        })
        .collect();

    c.bench_function("kpss_rw_2000_c_auto", |b| {
        b.iter(|| kpss(&rw, Regression::C, &NLags::Auto))
    });
    c.bench_function("kpss_rw_2000_ct_auto", |b| {
        b.iter(|| kpss(&rw, Regression::Ct, &NLags::Auto))
    });
    c.bench_function("kpss_rw_2000_c_legacy", |b| {
        b.iter(|| kpss(&rw, Regression::C, &NLags::Legacy))
    });
}

criterion_group!(benches, bench_kpss);
criterion_main!(benches);
