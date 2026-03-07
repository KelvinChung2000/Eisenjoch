//! PyO3 Python bindings for the nextpnr-rust FPGA place-and-route tool.
//!
//! Exposes the Rust implementation as `import nextpnr` in Python.

use pyo3::exceptions::{PyFileNotFoundError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use ::nextpnr::chipdb::ChipDb;
use ::nextpnr::context::Context;
use ::nextpnr::frontend::parse_json;
use ::nextpnr::placer::Placer;
use ::nextpnr::placer::heap::{PlacerHeap, PlacerHeapCfg};
use ::nextpnr::placer::sa::{PlacerSa, PlacerSaCfg};
use ::nextpnr::router::Router;
use ::nextpnr::router::router1::{Router1, Router1Cfg};
use ::nextpnr::router::router2::{Router2, Router2Cfg};
use ::nextpnr::timing::TimingAnalyser;
use ::nextpnr::types::DelayT;

use std::path::Path;

// ---------------------------------------------------------------------------
// PyContext
// ---------------------------------------------------------------------------

/// Main Context class exposed to Python.
///
/// Wraps the Rust `Context` (chip database + design netlist + placement/routing
/// state) and a `TimingAnalyser` for static timing analysis.
#[pyclass(name = "Context")]
pub struct PyContext {
    ctx: Context,
    timing: TimingAnalyser,
}

#[pymethods]
impl PyContext {
    /// Create a new context.
    ///
    /// Args:
    ///     chipdb: Path to a `.bin` chip database file.
    ///     device: Device name (not yet implemented).
    #[new]
    #[pyo3(signature = (*, chipdb=None, device=None))]
    fn new(chipdb: Option<&str>, device: Option<&str>) -> PyResult<Self> {
        let chipdb_path = match (chipdb, device) {
            (Some(path), _) => path.to_string(),
            (None, Some(_dev)) => {
                return Err(PyValueError::new_err(
                    "Device name lookup not yet implemented, use chipdb= parameter",
                ));
            }
            (None, None) => {
                return Err(PyValueError::new_err(
                    "Either chipdb or device must be specified",
                ));
            }
        };

        let db = ChipDb::load(Path::new(&chipdb_path)).map_err(|e| {
            PyFileNotFoundError::new_err(format!("Failed to load chipdb: {}", e))
        })?;

        Ok(Self {
            ctx: Context::new(db),
            timing: TimingAnalyser::new(),
        })
    }

    /// Load a Yosys JSON netlist into the design.
    ///
    /// Args:
    ///     path: Path to a Yosys JSON file.
    fn load_design(&mut self, path: &str) -> PyResult<()> {
        let json_str = std::fs::read_to_string(path).map_err(|e| {
            PyFileNotFoundError::new_err(format!("Failed to read {}: {}", path, e))
        })?;
        let design = parse_json(&json_str, &self.ctx.id_pool).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to parse JSON: {}", e))
        })?;
        self.ctx.design = design;
        Ok(())
    }

    /// Run the packer on the design.
    ///
    /// Args:
    ///     plugin: Path to a packer plugin (not yet implemented).
    #[pyo3(signature = (*, plugin=None))]
    fn pack(&mut self, plugin: Option<&str>) -> PyResult<()> {
        if plugin.is_some() {
            return Err(PyRuntimeError::new_err(
                "Plugin loading not yet implemented",
            ));
        }
        ::nextpnr::packer::pack(&mut self.ctx, None)
            .map_err(|e| PyRuntimeError::new_err(format!("Packer error: {}", e)))
    }

    /// Run a placer on the design.
    ///
    /// Args:
    ///     placer: Placer algorithm name ("heap" or "sa"). Default "heap".
    ///     seed: RNG seed for reproducibility. Default 1.
    #[pyo3(signature = (*, placer="heap", seed=1))]
    fn place(&mut self, placer: &str, seed: u64) -> PyResult<()> {
        match placer {
            "heap" => {
                let mut cfg = PlacerHeapCfg::default();
                cfg.seed = seed;
                PlacerHeap.place(&mut self.ctx, &cfg)
                    .map_err(|e| PyRuntimeError::new_err(format!("HeAP placer error: {}", e)))
            }
            "sa" => {
                let mut cfg = PlacerSaCfg::default();
                cfg.seed = seed;
                PlacerSa.place(&mut self.ctx, &cfg)
                    .map_err(|e| PyRuntimeError::new_err(format!("SA placer error: {}", e)))
            }
            _ => Err(PyValueError::new_err(format!(
                "Unknown placer: {}",
                placer
            ))),
        }
    }

    /// Run a router on the design.
    ///
    /// Args:
    ///     router: Router algorithm name ("router1" or "router2"). Default "router1".
    #[pyo3(signature = (*, router="router1"))]
    fn route(&mut self, router: &str) -> PyResult<()> {
        match router {
            "router1" => {
                let cfg = Router1Cfg::default();
                Router1.route(&mut self.ctx, &cfg)
                    .map_err(|e| PyRuntimeError::new_err(format!("Router1 error: {}", e)))
            }
            "router2" => {
                let cfg = Router2Cfg::default();
                Router2.route(&mut self.ctx, &cfg)
                    .map_err(|e| PyRuntimeError::new_err(format!("Router2 error: {}", e)))
            }
            _ => Err(PyValueError::new_err(format!(
                "Unknown router: {}",
                router
            ))),
        }
    }

    /// Add a clock constraint.
    ///
    /// Args:
    ///     net_name: Name of the clock net.
    ///     freq_mhz: Clock frequency in MHz.
    fn add_clock(&mut self, net_name: &str, freq_mhz: f64) -> PyResult<()> {
        let id = self.ctx.id_pool.intern(net_name);
        self.timing.add_clock_constraint(id, freq_mhz);
        // Also set clock_constraint on the net if it exists in the design.
        if let Some(net_idx) = self.ctx.design.net_by_name(id) {
            let period_ps = (1_000_000.0 / freq_mhz) as DelayT;
            self.ctx.design.net_edit(net_idx).set_clock_constraint(period_ps);
        }
        Ok(())
    }

    /// Run timing analysis and return a report.
    ///
    /// Returns:
    ///     A TimingReport with fmax, worst slack, and endpoint counts.
    fn timing_report(&mut self) -> PyResult<PyTimingReport> {
        self.timing.analyse(&self.ctx.design, &self.ctx.id_pool);
        let report = self.timing.report();
        Ok(PyTimingReport {
            fmax: report.fmax,
            worst_slack: report.worst_slack,
            num_failing: report.num_failing,
            num_endpoints: report.num_endpoints,
        })
    }

    /// Write the design to a JSON file (not yet implemented).
    fn write_design(&self, _path: &str) -> PyResult<()> {
        Err(PyRuntimeError::new_err("JSON writer not yet implemented"))
    }

    /// Grid width in tiles.
    #[getter]
    fn width(&self) -> i32 {
        self.ctx.chipdb().width()
    }

    /// Grid height in tiles.
    #[getter]
    fn height(&self) -> i32 {
        self.ctx.chipdb().height()
    }

    /// List of all cell names in the design.
    #[getter]
    fn cells(&self) -> Vec<String> {
        self.ctx
            .cells()
            .filter(|c| c.is_alive())
            .map(|c| c.name().to_string())
            .collect()
    }

    /// List of all net names in the design.
    #[getter]
    fn nets(&self) -> Vec<String> {
        self.ctx
            .nets()
            .filter(|n| n.is_alive())
            .map(|n| n.name().to_string())
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "Context(width={}, height={}, cells={}, nets={})",
            self.ctx.chipdb().width(),
            self.ctx.chipdb().height(),
            self.ctx.design.num_cells(),
            self.ctx.design.num_nets()
        )
    }
}

// ---------------------------------------------------------------------------
// PyTimingReport
// ---------------------------------------------------------------------------

/// Summary report of timing analysis results.
#[pyclass(name = "TimingReport")]
pub struct PyTimingReport {
    /// Maximum achievable frequency in MHz.
    #[pyo3(get)]
    pub fmax: f64,
    /// Worst negative slack in picoseconds.
    #[pyo3(get)]
    pub worst_slack: DelayT,
    /// Number of endpoints with negative slack (failing timing).
    #[pyo3(get)]
    pub num_failing: usize,
    /// Total number of timing endpoints analysed.
    #[pyo3(get)]
    pub num_endpoints: usize,
}

#[pymethods]
impl PyTimingReport {
    fn __repr__(&self) -> String {
        format!(
            "TimingReport(fmax={:.2} MHz, worst_slack={} ps, failing={}/{})",
            self.fmax, self.worst_slack, self.num_failing, self.num_endpoints
        )
    }
}

// ---------------------------------------------------------------------------
// Python module
// ---------------------------------------------------------------------------

/// The `nextpnr` Python module.
#[pymodule]
fn nextpnr(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyContext>()?;
    m.add_class::<PyTimingReport>()?;
    Ok(())
}
