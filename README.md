# rsomics-kpss-test

Kwiatkowski-Phillips-Schmidt-Shin (KPSS) stationarity test — a value-exact Rust reimplementation of
`statsmodels.tsa.stattools.kpss`.

```
rsomics-kpss-test [OPTIONS] [SERIES]
```

Reads one floating-point value per line from `SERIES` (or stdin with `-`).
Tests the null hypothesis that the series is level or trend stationary.
Small p-value → reject stationarity (unit root present).

## Usage

```
Options:
  --regression <TYPE>    c (level stationarity) or ct (trend stationarity)  [default: c]
  --nlags <METHOD>       auto (Hobijn 1998), legacy (Schwert 1989), or a fixed integer  [default: auto]
  --json                 Emit JSON envelope
```

## Performance

Measured on mini_m2 (aarch64-apple-darwin), 200-observation random walk, `--regression c --nlags auto`:

| | wall time |
|---|---|
| rsomics-kpss-test 0.1.0 | 2.1 ms |
| statsmodels 0.14.6 | 310 ms |

**148× end-to-end** (Rust pays process startup + file I/O; Python number excludes interpreter startup).

## Install

```
cargo install rsomics-kpss-test
```

## Origin

This crate is an independent Rust reimplementation of `statsmodels.tsa.stattools.kpss` based on:

- Kwiatkowski, D., Phillips, P.C.B., Schmidt, P. & Shin, Y. (1992). Testing the null hypothesis of stationarity against the alternative of a unit root. _Journal of Econometrics_ 54(1–3), 159–178.
- Hobijn, B., Franses, P.H. & Ooms, M. (1998). Generalizations of the KPSS-test for stationarity. _Econometric Institute Research Papers EI 9802_.
- The statsmodels 0.14.6 source (BSD-3-Clause) was read to ensure exact algorithmic compatibility (Bartlett HAC kernel, Hobijn autolag formula, critical-value interpolation table).

License: MIT OR Apache-2.0.
Upstream credit: [statsmodels](https://www.statsmodels.org/) (BSD-3-Clause).
