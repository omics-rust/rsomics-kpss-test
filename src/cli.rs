use std::io::{BufRead, stdin};
use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use rsomics_common::{CommonFlags, RsomicsError, ToolMeta, run};
use serde::Serialize;

use rsomics_kpss_test::{NLags, Regression, kpss};

pub const META: ToolMeta = ToolMeta {
    name: env!("CARGO_PKG_NAME"),
    version: env!("CARGO_PKG_VERSION"),
};

/// Kwiatkowski-Phillips-Schmidt-Shin (KPSS) stationarity test.
///
/// Tests the null hypothesis that the series is level or trend stationary.
/// Reads one floating-point value per line from SERIES (or stdin with `-`).
/// Small p-value → reject stationarity (unit root present).
#[derive(Parser, Debug)]
#[command(name = "rsomics-kpss-test", version, about, long_about = None)]
pub struct Cli {
    /// Input series: one float per line; `-` reads stdin.
    #[arg(value_name = "SERIES")]
    pub series: Option<PathBuf>,

    /// Regression type: c (level stationarity) or ct (trend stationarity).
    #[arg(long, default_value = "c", value_name = "TYPE")]
    pub regression: String,

    /// Lag selection: auto (Hobijn 1998), legacy (Schwert 1989), or an integer.
    #[arg(long, default_value = "auto", value_name = "METHOD")]
    pub nlags: String,

    #[command(flatten)]
    pub common: CommonFlags,
}

#[derive(Serialize)]
struct Output {
    kpss_stat: f64,
    pvalue: f64,
    lags: usize,
    crit_10pct: f64,
    crit_5pct: f64,
    crit_2_5pct: f64,
    crit_1pct: f64,
}

impl Cli {
    pub fn run(self) -> ExitCode {
        let common = self.common;
        run(&common, META, move || {
            let regression = Regression::parse(&self.regression).ok_or_else(|| {
                RsomicsError::InvalidInput(format!(
                    "unknown regression '{}'; expected c or ct",
                    self.regression
                ))
            })?;
            let nlags = NLags::parse(&self.nlags).ok_or_else(|| {
                RsomicsError::InvalidInput(format!(
                    "unknown nlags '{}'; expected auto, legacy, or a non-negative integer",
                    self.nlags
                ))
            })?;

            let x = read_series(self.series.as_ref())?;
            let result = kpss(&x, regression, &nlags);

            Ok(Output {
                kpss_stat: result.kpss_stat,
                pvalue: result.pvalue,
                lags: result.lags,
                crit_10pct: result.crit_10pct,
                crit_5pct: result.crit_5pct,
                crit_2_5pct: result.crit_2_5pct,
                crit_1pct: result.crit_1pct,
            })
        })
    }
}

fn read_series(path: Option<&PathBuf>) -> rsomics_common::Result<Vec<f64>> {
    let reader: Box<dyn BufRead> = match path {
        None => Box::new(stdin().lock()),
        Some(p) if p.as_os_str() == "-" => Box::new(stdin().lock()),
        Some(p) => Box::new(std::io::BufReader::new(
            std::fs::File::open(p).map_err(RsomicsError::Io)?,
        )),
    };

    let mut values = Vec::new();
    for line in reader.lines() {
        let line = line.map_err(RsomicsError::Io)?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let v: f64 = fast_float2::parse(trimmed)
            .map_err(|_| RsomicsError::InvalidInput(format!("cannot parse float: '{trimmed}'")))?;
        values.push(v);
    }
    Ok(values)
}

#[test]
fn cli_debug_assert() {
    use clap::CommandFactory;
    Cli::command().debug_assert();
}
