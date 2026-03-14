//! PyO3 Python bindings for the nextpnr-rust FPGA place-and-route tool.
//!
//! Exposes the Rust implementation as `import nextpnr` in Python.

use pyo3::exceptions::{PyFileNotFoundError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use ::nextpnr::checkpoint;
use ::nextpnr::chipdb::ChipDb;
use ::nextpnr::context::Context;
use ::nextpnr::frontend::parse_json;
use ::nextpnr::netlist::Rect;
use ::nextpnr::placer::electro_place::{ElectroPlaceCfg, PlacerElectro};
use ::nextpnr::placer::heap::{PlacerHeap, PlacerHeapCfg};
use ::nextpnr::placer::hydraulic_place::{HydraulicPlacerCfg, PlacerHydraulic};
use ::nextpnr::placer::sa::{PlacerSa, PlacerSaCfg};
use ::nextpnr::placer::Placer;
use ::nextpnr::router::router1::{Router1, Router1Cfg};
use ::nextpnr::router::router2::{Router2, Router2Cfg};
use ::nextpnr::router::Router;
use ::nextpnr::timing::{DelayT, TimingAnalyser};
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

        let db = ChipDb::load(Path::new(&chipdb_path))
            .map_err(|e| PyFileNotFoundError::new_err(format!("Failed to load chipdb: {}", e)))?;

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
        let json_str = std::fs::read_to_string(path)
            .map_err(|e| PyFileNotFoundError::new_err(format!("Failed to read {}: {}", path, e)))?;
        let design = parse_json(&json_str, &self.ctx.id_pool)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to parse JSON: {}", e)))?;
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

    /// Add a region constraint.
    ///
    /// Args:
    ///     name: Region name.
    ///     rects: List of [x0, y0, x1, y1] rectangles.
    fn add_region(&mut self, name: &str, rects: Vec<[i32; 4]>) -> PyResult<u32> {
        let id = self.ctx.id_pool.intern(name);
        let idx = self.ctx.design.add_region(id);
        let region = self.ctx.design.region_mut(idx);
        for r in rects {
            region.rects.push(Rect::new(r[0], r[1], r[2], r[3]));
        }
        self.ctx.invalidate_region_cache();
        Ok(idx)
    }

    /// Constrain a cell to a region.
    ///
    /// Args:
    ///     cell: Cell name.
    ///     region: Region index (from add_region).
    fn constrain_cell_to_region(&mut self, cell: &str, region: u32) -> PyResult<()> {
        let cell_id = self.ctx.id_pool.intern(cell);
        let cell_idx = self
            .ctx
            .design
            .cell_by_name(cell_id)
            .ok_or_else(|| PyValueError::new_err(format!("Cell not found: {}", cell)))?;
        self.ctx.design.cell_edit(cell_idx).set_region(Some(region));
        Ok(())
    }

    /// Constrain multiple cells to a region.
    fn constrain_cells_to_region(&mut self, cells: Vec<String>, region: u32) -> PyResult<()> {
        for cell in &cells {
            self.constrain_cell_to_region(cell, region)?;
        }
        Ok(())
    }

    /// Save a checkpoint of current placement/routing state.
    fn save_checkpoint(&self, path: &str) -> PyResult<()> {
        checkpoint::save(&self.ctx, Path::new(path))
            .map_err(|e| PyRuntimeError::new_err(format!("Save checkpoint error: {}", e)))
    }

    /// Load a checkpoint and restore placements/routes.
    ///
    /// Restored cells are placed as Fixed, so subsequent place()/route()
    /// calls will skip them and only handle new/changed cells and nets.
    fn load_checkpoint(&mut self, path: &str) -> PyResult<()> {
        let cp = checkpoint::Checkpoint::load_from_file(Path::new(path))
            .map_err(|e| PyRuntimeError::new_err(format!("Load checkpoint error: {}", e)))?;
        checkpoint::restore(&mut self.ctx, &cp)
            .map_err(|e| PyRuntimeError::new_err(format!("Restore error: {}", e)))?;
        Ok(())
    }

    /// Lock a cell to a specific BEL before placement.
    ///
    /// The BEL is identified by tile coordinates (x, y) and a BEL name
    /// (e.g. "IO0", "L0_LUT", "L0_FF").
    ///
    /// Args:
    ///     cell: Cell name.
    ///     x: Tile X coordinate.
    ///     y: Tile Y coordinate.
    ///     bel_name: Name of the BEL within the tile.
    fn place_cell(&mut self, cell: &str, x: i32, y: i32, bel_name: &str) -> PyResult<()> {
        use ::nextpnr::chipdb::BelId;
        use ::nextpnr::common::PlaceStrength;

        let cell_id = self.ctx.id_pool.intern(cell);
        let cell_idx = self
            .ctx
            .design
            .cell_by_name(cell_id)
            .ok_or_else(|| PyValueError::new_err(format!("Cell '{}' not found", cell)))?;

        let tile = self.ctx.chipdb().tile_by_xy(x, y);
        let chipdb = self.ctx.chipdb();
        let tt = chipdb.tile_type(tile);
        let bel_idx = tt
            .bels
            .get()
            .iter()
            .position(|b| {
                let n: i32 = unsafe { ::nextpnr::read_packed!(*b, name) };
                chipdb.constid_str(n) == Some(bel_name)
            })
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "BEL '{}' not found in tile ({}, {})",
                    bel_name, x, y
                ))
            })?;

        let bel = BelId::new(tile, bel_idx as i32);
        if !self.ctx.bind_bel(bel, cell_idx, PlaceStrength::Locked) {
            return Err(PyRuntimeError::new_err(format!(
                "Failed to bind cell '{}' to BEL '{}' at ({}, {})",
                cell, bel_name, x, y
            )));
        }
        Ok(())
    }

    /// Run a placer on the design.
    ///
    /// Args:
    ///     placer: Placer algorithm name ("heap", "sa", "hydraulic", or "electro"). Default "heap".
    ///     seed: RNG seed for reproducibility. Default 1.
    ///     max_iters: Maximum iterations (default varies by placer).
    ///     congestion_weight: Weight for congestion cost. Default 0.5.
    ///     turbulence_beta: Nonlinear resistance coefficient for hydraulic placer. Default 4.0.
    ///     newton_iters: Newton iterations for nonlinear resistance (hydraulic). Default 2.
    #[pyo3(signature = (*, placer="heap", seed=1, max_iters=None, congestion_weight=0.5, turbulence_beta=4.0, newton_iters=2, star_weight=1.0, pressure_weight_start=0.0, pressure_weight_end=2.0, io_boost=4.0, nesterov_step_size=0.1, wl_coeff=0.5, momentum=None))]
    fn place(
        &mut self,
        placer: &str,
        seed: u64,
        max_iters: Option<usize>,
        congestion_weight: f64,
        turbulence_beta: f64,
        newton_iters: usize,
        star_weight: f64,
        pressure_weight_start: f64,
        pressure_weight_end: f64,
        io_boost: f64,
        nesterov_step_size: f64,
        wl_coeff: f64,
        momentum: Option<f64>,
    ) -> PyResult<()> {
        match placer {
            "heap" => {
                let mut cfg = PlacerHeapCfg::default();
                cfg.seed = seed;
                cfg.congestion_weight = congestion_weight;
                PlacerHeap
                    .place(&mut self.ctx, &cfg)
                    .map_err(|e| PyRuntimeError::new_err(format!("HeAP placer error: {}", e)))
            }
            "sa" => {
                let mut cfg = PlacerSaCfg::default();
                cfg.seed = seed;
                cfg.congestion_weight = congestion_weight;
                PlacerSa
                    .place(&mut self.ctx, &cfg)
                    .map_err(|e| PyRuntimeError::new_err(format!("SA placer error: {}", e)))
            }
            "hydraulic" => {
                let mut cfg = HydraulicPlacerCfg::default();
                cfg.seed = seed;
                cfg.turbulence_beta = turbulence_beta;
                cfg.newton_iters = newton_iters;
                cfg.star_weight = star_weight;
                cfg.pressure_weight_start = pressure_weight_start;
                cfg.pressure_weight_end = pressure_weight_end;
                cfg.io_boost = io_boost;
                cfg.nesterov_step_size = nesterov_step_size;
                cfg.wl_coeff = wl_coeff;
                cfg.momentum = momentum;
                if let Some(iters) = max_iters {
                    cfg.max_outer_iters = iters;
                }
                PlacerHydraulic
                    .place(&mut self.ctx, &cfg)
                    .map_err(|e| PyRuntimeError::new_err(format!("Hydraulic placer error: {}", e)))
            }
            "electro" => {
                let mut cfg = ElectroPlaceCfg::default();
                cfg.seed = seed;
                if let Some(iters) = max_iters {
                    cfg.max_iters = iters;
                }
                PlacerElectro
                    .place(&mut self.ctx, &cfg)
                    .map_err(|e| PyRuntimeError::new_err(format!("ElectroPlace error: {}", e)))
            }
            _ => Err(PyValueError::new_err(format!(
                "Unknown placer: {}. Available: heap, sa, hydraulic, electro",
                placer
            ))),
        }
    }

    /// Run a router on the design.
    ///
    /// Args:
    ///     router: Router algorithm name ("router1" or "router2"). Default "router1".
    ///     bb_margin: Bounding box margin for Router2 (tiles). Default 3.
    ///     max_iterations: Max routing iterations. Router1 default 500, Router2 default 50.
    #[pyo3(signature = (*, router="router1", bb_margin=None, max_iterations=None))]
    fn route(
        &mut self,
        router: &str,
        bb_margin: Option<i32>,
        max_iterations: Option<usize>,
    ) -> PyResult<()> {
        match router {
            "router1" => {
                let mut cfg = Router1Cfg::default();
                if let Some(iters) = max_iterations {
                    cfg.max_iterations = iters;
                }
                Router1
                    .route(&mut self.ctx, &cfg)
                    .map_err(|e| PyRuntimeError::new_err(format!("Router1 error: {}", e)))
            }
            "router2" => {
                let mut cfg = Router2Cfg::default();
                if let Some(margin) = bb_margin {
                    cfg.bb_margin = margin;
                }
                if let Some(iters) = max_iterations {
                    cfg.max_iterations = iters;
                }
                Router2
                    .route(&mut self.ctx, &cfg)
                    .map_err(|e| PyRuntimeError::new_err(format!("Router2 error: {}", e)))
            }
            _ => Err(PyValueError::new_err(format!("Unknown router: {}", router))),
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
        if let Some(net_idx) = self.ctx.design.net_by_name(id) {
            let period_ps = (1_000_000.0 / freq_mhz) as DelayT;
            self.ctx
                .design
                .net_edit(net_idx)
                .set_clock_constraint(period_ps);
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

    /// Generate a resource utilization report.
    ///
    /// Returns:
    ///     A UtilizationReport with per-resource-type usage.
    fn utilization_report(&self) -> PyUtilizationReport {
        let report = self.ctx.utilization_report();
        let rows = report
            .rows
            .iter()
            .map(|r| (r.resource.clone(), r.used, r.available, r.percent()))
            .collect();
        PyUtilizationReport {
            rows,
            total_cells: report.total_cells,
            total_nets: report.total_nets,
            placed_cells: report.placed_cells,
            text: report.to_string(),
        }
    }

    /// Compute spatial placement density using a sliding window.
    ///
    /// Divides the chip into regions of `window` x `window` tiles and computes
    /// the fraction of BELs occupied in each region. Returns a dict with:
    ///   - max_density: highest regional density (0.0-1.0)
    ///   - avg_density: average regional density
    ///   - hotspot: (x, y) tile coords of the densest region's top-left corner
    ///   - hot_regions: count of regions above 50% density
    ///   - grid: list of (x, y, density) for regions above 50%
    #[pyo3(signature = (window=10))]
    fn placement_density(&self, py: Python<'_>, window: i32) -> PyResult<PyObject> {
        let report = self.ctx.placement_density(window);
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("max_density", report.max_density)?;
        dict.set_item("avg_density", report.avg_density)?;
        dict.set_item("hotspot", report.hotspot)?;
        dict.set_item("hot_regions", report.hot_regions)?;
        dict.set_item("grid", report.grid)?;
        Ok(dict.into())
    }

    /// Estimate routing congestion.
    ///
    /// Returns a dict with max_congestion, avg_congestion, hotspot, hotspot_axis,
    /// and hot_edges above the given threshold.
    #[pyo3(signature = (threshold=0.5))]
    fn congestion_estimate(&self, py: Python<'_>, threshold: f64) -> PyResult<PyObject> {
        let report = self.ctx.estimate_congestion(threshold);
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("max_congestion", report.max_congestion)?;
        dict.set_item("avg_congestion", report.avg_congestion)?;
        dict.set_item("hotspot", report.hotspot)?;
        dict.set_item("hotspot_axis", format!("{:?}", report.hotspot_axis))?;
        let hot_edges: Vec<(i32, i32, String, f64)> = report
            .hot_edges
            .iter()
            .map(|(x, y, axis, c)| (*x, *y, format!("{:?}", axis), *c))
            .collect();
        dict.set_item("hot_edges", hot_edges)?;
        Ok(dict.into())
    }

    fn total_hpwl(&self) -> f64 {
        ::nextpnr::metrics::total_hpwl(&self.ctx)
    }

    /// Total Bresenham line estimate wirelength (tighter than HPWL).
    fn total_line_estimate(&self) -> f64 {
        ::nextpnr::metrics::total_line_estimate(&self.ctx)
    }

    /// Total routed wirelength (wire count). Only meaningful after routing.
    fn total_routed_wirelength(&self) -> usize {
        ::nextpnr::metrics::total_routed_wirelength(&self.ctx)
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
// PyUtilizationReport
// ---------------------------------------------------------------------------

/// Resource utilization report.
#[pyclass(name = "UtilizationReport")]
pub struct PyUtilizationReport {
    rows: Vec<(String, usize, usize, f64)>,
    #[pyo3(get)]
    pub total_cells: usize,
    #[pyo3(get)]
    pub total_nets: usize,
    #[pyo3(get)]
    pub placed_cells: usize,
    #[pyo3(get)]
    pub text: String,
}

#[pymethods]
impl PyUtilizationReport {
    /// List of resource rows as (resource, used, available, percent) tuples.
    #[getter]
    fn rows(&self) -> Vec<(String, usize, usize, f64)> {
        self.rows.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "UtilizationReport(cells={}, nets={}, placed={})",
            self.total_cells, self.total_nets, self.placed_cells
        )
    }

    fn __str__(&self) -> &str {
        &self.text
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
    m.add_class::<PyUtilizationReport>()?;
    Ok(())
}
