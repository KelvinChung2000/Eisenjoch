//! PyO3 Python bindings for the nextpnr-rust FPGA place-and-route tool.
//!
//! Exposes the Rust implementation as `import nextpnr` in Python.

use pyo3::exceptions::{PyFileNotFoundError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use ::nextpnr::checkpoint;
use ::nextpnr::fasm;
use ::nextpnr::routed_json::routed_json;
use ::nextpnr::chipdb::ChipDb;
use ::nextpnr::context::Context;
use ::nextpnr::frontend::parse_json;
use ::nextpnr::netlist::{NetId, Rect};
use ::nextpnr::placer::electro_place::{ElectroPlaceCfg, PlacerElectro};
use ::nextpnr::placer::heap::{PlacerHeap, PlacerHeapCfg};
use ::nextpnr::placer::opt_trans::config::InitStrategy;
use ::nextpnr::placer::opt_trans::{OptTransPlacerCfg, PlacerOptTrans};
use ::nextpnr::placer::sa::{PlacerSa, PlacerSaCfg};
use ::nextpnr::placer::{refine_placement, Placer, PlacerPipeline, RefineCfg};
use ::nextpnr::router::maze::{Router1, Router1Cfg};
use ::nextpnr::router::raster::{RasterRouter, RasterRouterCfg};
use ::nextpnr::router::router2::{Router2, Router2Cfg, TimingDrivenCfg};
use ::nextpnr::router::Router;
use ::nextpnr::common::{ClockEdge, DelayT};
use ::nextpnr::sta::{
    register_clock_nets, routed_back_annotation, BackAnnotation, CritNorm, DelayBounds, Criticality, InterconnectModel, InterconnectStage, PlacedEstimate, RustSta, StaSession,
    StaSessionError, NOTIONAL_CLASSLESS_PIPS,
};
use nextpnr_upstream::{
    Nextpnr, Placer as UpstreamPlacer, Request, Router as UpstreamRouter, Stage,
};
use rustc_hash::FxHashMap;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// PyContext
// ---------------------------------------------------------------------------

/// Main Context class exposed to Python.
///
/// Wraps the Rust `Context` (chip database + design netlist + placement/routing
/// state).
#[pyclass(name = "Context")]
pub struct PyContext {
    ctx: Context,
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

    /// Register a cell type alias for BEL bucket mapping.
    ///
    /// Cells of `cell_type` will be placed on BELs of `bel_type`.
    /// Use this for architectures where netlist cell types differ from chipdb BEL types.
    fn add_cell_type_alias(&mut self, cell_type: &str, bel_type: &str) {
        self.ctx.add_cell_type_alias(cell_type, bel_type);
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
    ///     placer: Placer algorithm name ("heap", "sa", "opt_trans", or "electro"). Default "heap".
    ///     seed: RNG seed for reproducibility. Default 1.
    ///     max_iters: Maximum iterations (default varies by placer).
    ///     congestion_weight: Weight for congestion cost (HeAP/SA). Default 1.0.
    ///     timing_weight: Timing-driven weight, opt_trans only; criticality from RustSTA on ideal wires. Default 0.0.
    ///     init_strategy: Cell init strategy for opt_trans ("random_bel", "centroid", "uniform", "topological"). Default "topological" — centroid is degenerate on designs with few/colocated fixed cells.
    ///     num_threads: Rayon worker thread count for opt_trans solves. Default 8.
    #[pyo3(signature = (*, placer="heap", seed=1, max_iters=None, congestion_weight=1.0, timing_weight=0.0, init_strategy="topological", num_threads=8, legalization="ring"))]
    fn place(
        &mut self,
        placer: &str,
        seed: u64,
        max_iters: Option<usize>,
        congestion_weight: f64,
        timing_weight: f64,
        init_strategy: &str,
        num_threads: usize,
        legalization: &str,
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
            "opt_trans" | "hydraulic" => {
                let mut cfg = OptTransPlacerCfg::default();
                cfg.seed = seed;
                cfg.timing_weight = timing_weight;
                cfg.num_threads = num_threads.max(1);
                cfg.init_strategy = match init_strategy {
                    "centroid" => InitStrategy::Centroid,
                    "uniform" => InitStrategy::Uniform,
                    "random_bel" => InitStrategy::RandomBel,
                    "topological" | "topo" => InitStrategy::Topological,
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "Unknown init_strategy: {}. Available: centroid, uniform, random_bel, topological",
                            other
                        )))
                    }
                };
                cfg.legalization = legalization.to_string();
                if cfg.timing_weight > 0.0 {
                    cfg.timing_criticality = ideal_net_criticality(&self.ctx).map_err(sta_error)?;
                }
                if let Some(iters) = max_iters {
                    cfg.max_outer_iters = iters;
                }
                PlacerOptTrans
                    .place(&mut self.ctx, &cfg)
                    .map_err(|e| PyRuntimeError::new_err(format!("OptTrans placer error: {}", e)))
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
                "Unknown placer: {}. Available: heap, sa, opt_trans, hydraulic, electro",
                placer
            ))),
        }
    }

    /// Run placer1 refinement on a legal placement; raises RuntimeError if it fails or leaves it illegal.
    ///
    /// Args:
    ///     timing_driven: Weight moves by RustSTA criticality on placed-wire estimates.
    ///     crit_norm: "required_time" (1 - s/D_req) or "wns_relaxed" (required times shifted by a failing WNS).
    #[pyo3(signature = (*, timing_driven, crit_norm))]
    fn refine(&mut self, timing_driven: bool, crit_norm: &str) -> PyResult<()> {
        let crit_norm = match crit_norm {
            "required_time" => CritNorm::RequiredTime,
            "wns_relaxed" => CritNorm::WnsRelaxed,
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unknown crit_norm: {other}. Available: required_time, wns_relaxed"
                )))
            }
        };
        let cfg = RefineCfg {
            timing_driven,
            crit_norm,
            ..RefineCfg::default()
        };
        refine_placement(&mut self.ctx, &cfg)
            .and_then(|_| PlacerPipeline::validate(&self.ctx))
            .map_err(|e| PyRuntimeError::new_err(format!("Refine placer error: {}", e)))
    }

    /// Run a router on the design.
    ///
    /// Args:
    ///     router: Router algorithm name ("router1", "router2", or "raster"). Default "router1".
    ///     bb_margin: Bounding box margin in tiles for A* pruning. Default 3.
    ///     max_iterations: Max routing iterations. Router1 default 500, Router2 default 50.
    ///     skip_unplaced: If true, skip nets whose driver cell is not placed. Default false.
    ///     timing_driven: Weight each connection's delay by its STA criticality (router2 only). Default false.
    #[pyo3(signature = (*, router="router1", bb_margin=None, max_iterations=None, skip_unplaced=false, timing_driven=false))]
    fn route(
        &mut self,
        router: &str,
        bb_margin: Option<i32>,
        max_iterations: Option<usize>,
        skip_unplaced: bool,
        timing_driven: bool,
    ) -> PyResult<()> {
        if timing_driven && router != "router2" {
            return Err(PyValueError::new_err(format!(
                "Router {router} is not timing-driven; only router2 is"
            )));
        }
        use ::nextpnr::router::common::collect_routable_nets;

        // Collect nets, optionally filtering out those touching unplaced
        // or IO/packer cells (whose routing may not be modeled).
        let nets = if skip_unplaced {
            collect_routable_nets(&self.ctx)
                .into_iter()
                .filter(|&net_idx| {
                    let net = self.ctx.net(net_idx);
                    // Skip nets driven by unplaced or IO/packer cells
                    if let Some(dp) = net.driver_cell_port() {
                        let cell = self.ctx.cell(dp.cell);
                        if cell.bel().is_none() {
                            return false;
                        }
                        let name_str = self.ctx.name_of(cell.name_id());
                        if name_str.starts_with("$io$") || name_str.starts_with("$PACKER_") {
                            return false;
                        }
                    }
                    // Skip nets with any unplaced or IO/packer sink cell
                    net.users().iter().all(|user| {
                        if !user.is_valid() {
                            return true;
                        }
                        let cell = self.ctx.cell(user.cell);
                        if cell.bel().is_none() {
                            return false;
                        }
                        let name_str = self.ctx.name_of(cell.name_id());
                        !name_str.starts_with("$io$") && !name_str.starts_with("$PACKER_")
                    })
                })
                .collect::<Vec<_>>()
        } else {
            collect_routable_nets(&self.ctx)
        };

        match router {
            "router1" => {
                let mut cfg = Router1Cfg::default();
                if let Some(margin) = bb_margin {
                    cfg.bb_margin = margin;
                }
                if let Some(iters) = max_iterations {
                    cfg.max_iterations = iters;
                }
                Router1
                    .route_nets(&mut self.ctx, &cfg, &nets)
                    .map_err(|e| PyRuntimeError::new_err(format!("Router1 error: {}", e)))
            }
            "router2" => {
                let mut cfg = Router2Cfg {
                    timing: timing_driven.then(TimingDrivenCfg::default),
                    ..Router2Cfg::default()
                };
                if let Some(margin) = bb_margin {
                    cfg.bb_margin = margin;
                }
                if let Some(iters) = max_iterations {
                    cfg.max_iterations = iters;
                }
                Router2
                    .route_nets(&mut self.ctx, &cfg, &nets)
                    .map_err(|e| PyRuntimeError::new_err(format!("Router2 error: {}", e)))
            }
            "raster" => {
                let mut cfg = RasterRouterCfg::default();
                if let Some(iters) = max_iterations {
                    cfg.max_iterations = iters;
                }
                cfg.verbose = true;
                RasterRouter
                    .route_nets(&mut self.ctx, &cfg, &nets)
                    .map_err(|e| PyRuntimeError::new_err(format!("RasterRouter error: {}", e)))
            }
            "greedy" => {
                let mut cfg = RasterRouterCfg::default();
                cfg.use_greedy = true;
                cfg.max_iterations = 1;
                if let Some(iters) = max_iterations {
                    cfg.max_iterations = iters;
                }
                RasterRouter
                    .route_nets(&mut self.ctx, &cfg, &nets)
                    .map_err(|e| PyRuntimeError::new_err(format!("RasterRouter error: {}", e)))
            }
            _ => Err(PyValueError::new_err(format!("Unknown router: {}", router))),
        }
    }

    /// Constrain a clock net to a frequency; raises ValueError for an unknown net.
    ///
    /// Args:
    ///     net_name: Name of the clock net.
    ///     freq_mhz: Clock frequency in MHz.
    #[pyo3(signature = (net_name, *, freq_mhz))]
    fn add_clock(&mut self, net_name: &str, freq_mhz: f64) -> PyResult<()> {
        let period_ps = clock_period_ps(freq_mhz)?;
        let net = self
            .ctx
            .design
            .net_by_name(self.ctx.id_pool.intern(net_name))
            .ok_or_else(|| PyValueError::new_err(format!("Unknown clock net: {net_name}")))?;
        self.ctx.design.net_edit(net).set_clock_constraint(period_ps);
        Ok(())
    }

    /// Constrain every unconstrained net that drives a register clock pin; returns their names.
    ///
    /// Raises ValueError when the packed netlist has no unconstrained register clock net.
    #[pyo3(signature = (*, freq_mhz))]
    fn constrain_clocks(&mut self, freq_mhz: f64) -> PyResult<Vec<String>> {
        let period_ps = clock_period_ps(freq_mhz)?;
        let nets: Vec<NetId> = register_clock_nets(&self.ctx)
            .map_err(|e| sta_error(StaSessionError::from(e)))?
            .into_iter()
            .filter(|&net| self.ctx.design.net(net).clock_constraint == 0)
            .collect();
        if nets.is_empty() {
            return Err(PyValueError::new_err(
                "No unconstrained net drives a register clock pin; nothing to constrain",
            ));
        }
        for &net in &nets {
            self.ctx.design.net_edit(net).set_clock_constraint(period_ps);
        }
        Ok(nets
            .iter()
            .map(|&net| self.ctx.name_of(self.ctx.design.net(net).name).to_owned())
            .collect())
    }

    /// Append SDC commands (times in ns) and check them against the current netlist.
    ///
    /// Raises ValueError, leaving the constraints unchanged, when RustSTA rejects the text.
    #[pyo3(signature = (*, sdc))]
    fn apply_sdc(&mut self, sdc: &str) -> PyResult<()> {
        let accepted = self.ctx.design.sdc.len();
        self.ctx.design.sdc.push_str(sdc);
        self.ctx.design.sdc.push('\n');
        RustSta::build(&self.ctx).map(|_| ()).map_err(|e| {
            self.ctx.design.sdc.truncate(accepted);
            PyValueError::new_err(format!("Timing constraints rejected: {e}"))
        })
    }

    /// Run static timing analysis and return a report.
    ///
    /// Args:
    ///     stage: Interconnect delay source, "placed" (estimate) or "routed" (pip delays).
    ///     paths_per_domain: Worst paths kept per clock domain pair. Default 1.
    #[pyo3(signature = (*, stage, paths_per_domain=1))]
    fn timing_report(&self, stage: &str, paths_per_domain: usize) -> PyResult<PyTimingReport> {
        let stage = match stage {
            "placed" => InterconnectStage::Placed,
            "routed" => InterconnectStage::Routed,
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unknown timing stage: {other}. Available: placed, routed"
                )))
            }
        };
        let model = InterconnectModel::new(&self.ctx, NOTIONAL_CLASSLESS_PIPS, PlacedEstimate::ExactSearch)
            .map_err(|e| sta_error(StaSessionError::from(e)))?;
        let session = StaSession::build(&self.ctx, model, stage).map_err(sta_error)?;
        let report = session.report(&self.ctx, paths_per_domain).map_err(sta_error)?;
        Ok(PyTimingReport {
            fmax_mhz: report.fmax_mhz(),
            wns_ns: report.wns_ns(),
            tns_ns: report.tns_ns(),
            num_failing: report.failing_endpoints(),
            num_endpoints: report.endpoints(),
            hold_wns_ns: report.hold_wns_ns(),
            hold_tns_ns: report.hold_tns_ns(),
            num_hold_failing: report.hold_failing_endpoints(),
            num_hold_endpoints: report.hold_endpoints(),
            text: report.to_string(),
        })
    }

    /// Routed cell and interconnect delays in ps, for writing an SDF file.
    ///
    /// Raises RuntimeError when a net is not fully routed or the chipdb has no speed grade.
    fn routed_back_annotation(&self) -> PyResult<PyBackAnnotation> {
        let model = InterconnectModel::new(&self.ctx, NOTIONAL_CLASSLESS_PIPS, PlacedEstimate::ExactSearch)
            .map_err(|e| sta_error(StaSessionError::from(e)))?;
        routed_back_annotation(&self.ctx, &model)
            .map(PyBackAnnotation::from)
            .map_err(sta_error)
    }

    /// FASM features of the routed design, named by the chipdb's pip and BEL data.
    ///
    /// Raises RuntimeError when the chipdb lacks FASM data, a cell is unplaced or a parameter has no exact bits.
    fn fasm_features(&self) -> PyResult<Vec<PyFasmFeature>> {
        fasm::fasm_features(&self.ctx)
            .map(|features| features.into_iter().map(PyFasmFeature::from).collect())
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    /// The routed design as nextpnr `write_json` text, with NEXTPNR_BEL on cells and ROUTING on nets.
    ///
    /// Raises RuntimeError when a property is an integer wider than 32 bits.
    fn routed_json(&self) -> PyResult<String> {
        let json = routed_json(&self.ctx).map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(serde_json::to_string_pretty(&json).expect("a JSON value serialises"))
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
    fn placement_density<'py>(
        &self,
        py: Python<'py>,
        window: i32,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let report = self.ctx.placement_density(window);
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("max_density", report.max_density)?;
        dict.set_item("avg_density", report.avg_density)?;
        dict.set_item("hotspot", report.hotspot)?;
        dict.set_item("hot_regions", report.hot_regions)?;
        dict.set_item("grid", report.grid)?;
        Ok(dict)
    }

    /// Estimate routing congestion.
    ///
    /// Returns a dict with max_congestion, avg_congestion, hotspot, hotspot_axis,
    /// and hot_edges above the given threshold.
    #[pyo3(signature = (threshold=0.5))]
    fn congestion_estimate<'py>(
        &self,
        py: Python<'py>,
        threshold: f64,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
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
        Ok(dict)
    }

    fn total_hpwl(&self) -> f64 {
        ::nextpnr::metrics::total_hpwl(&self.ctx)
    }

    /// Ideal-clustering HPWL lower bound: sum over nets of the bbox of the
    /// net's locked (IO-fixed) cells only. Strict lower bound under any
    /// placement of movable cells.
    fn total_hpwl_locked_only(&self) -> f64 {
        ::nextpnr::metrics::total_hpwl_locked_only(&self.ctx)
    }

    /// Total Bresenham line estimate wirelength (tighter than HPWL).
    fn total_line_estimate(&self) -> f64 {
        ::nextpnr::metrics::total_line_estimate(&self.ctx)
    }

    /// Total congestion cost: Σ (demand/capacity)² across all tile edges.
    fn total_congestion_cost(&self) -> f64 {
        ::nextpnr::metrics::total_congestion_cost(&self.ctx)
    }

    /// Total routed wirelength (wire count). Only meaningful after routing.
    fn total_routed_wirelength(&self) -> usize {
        ::nextpnr::metrics::total_routed_wirelength(&self.ctx)
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

/// Setup and hold summary over all clock domain pairs; `str()` gives the per-pair report.
#[pyclass(name = "TimingReport")]
pub struct PyTimingReport {
    /// Lowest same-clock Fmax in MHz, or None without a same-clock domain pair.
    #[pyo3(get)]
    pub fmax_mhz: Option<f64>,
    /// Worst setup slack in ns, or None without a constrained endpoint.
    #[pyo3(get)]
    pub wns_ns: Option<f64>,
    /// Total negative setup slack in ns.
    #[pyo3(get)]
    pub tns_ns: f64,
    /// Endpoints with negative setup slack, summed over domain pairs.
    #[pyo3(get)]
    pub num_failing: usize,
    /// Constrained setup endpoints, summed over domain pairs.
    #[pyo3(get)]
    pub num_endpoints: usize,
    /// Worst hold slack in ns, or None without a hold check.
    #[pyo3(get)]
    pub hold_wns_ns: Option<f64>,
    /// Total negative hold slack in ns.
    #[pyo3(get)]
    pub hold_tns_ns: f64,
    /// Endpoints with negative hold slack, summed over domain pairs.
    #[pyo3(get)]
    pub num_hold_failing: usize,
    /// Hold-checked endpoints, summed over domain pairs.
    #[pyo3(get)]
    pub num_hold_endpoints: usize,
    text: String,
}

#[pymethods]
impl PyTimingReport {
    fn __repr__(&self) -> String {
        let optional = |value: Option<f64>| value.map_or("None".to_owned(), |v| format!("{v:.3}"));
        format!(
            "TimingReport(fmax_mhz={}, wns_ns={}, tns_ns={:.3}, failing={}/{}, hold_wns_ns={}, hold_failing={}/{})",
            optional(self.fmax_mhz),
            optional(self.wns_ns),
            self.tns_ns,
            self.num_failing,
            self.num_endpoints,
            optional(self.hold_wns_ns),
            self.num_hold_failing,
            self.num_hold_endpoints
        )
    }

    fn __str__(&self) -> String {
        self.text.clone()
    }
}

const PS_PER_US: f64 = 1_000_000.0;

fn clock_period_ps(freq_mhz: f64) -> PyResult<DelayT> {
    if !(freq_mhz.is_finite() && freq_mhz > 0.0) {
        return Err(PyValueError::new_err(format!(
            "Clock frequency must be positive and finite, got {freq_mhz} MHz"
        )));
    }
    Ok((PS_PER_US / freq_mhz).round() as DelayT)
}

fn sta_error(error: StaSessionError) -> PyErr {
    PyRuntimeError::new_err(format!("Timing analysis error: {error}"))
}

fn ideal_net_criticality(ctx: &Context) -> Result<FxHashMap<NetId, f32>, StaSessionError> {
    let model = InterconnectModel::new(ctx, NOTIONAL_CLASSLESS_PIPS, PlacedEstimate::ExactSearch)?;
    let session = StaSession::build_ideal(ctx, model)?;
    ctx.design
        .iter_alive_nets()
        .map(|(net, _)| {
            let criticality = session
                .net_sinks(net)?
                .iter()
                .filter_map(|sink| sink.criticality())
                .map(Criticality::value)
                .fold(0.0, f64::max);
            Ok((net, criticality as f32))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Back-annotation records
// ---------------------------------------------------------------------------

/// One INTERCONNECT delay from a driver pin to a load pin.
#[pyclass(name = "InterconnectDelay", get_all, frozen)]
pub struct PyInterconnectDelay {
    pub source_instance: String,
    pub source_port: String,
    pub load_instance: String,
    pub load_port: String,
    pub early_ps: DelayT,
    pub late_ps: DelayT,
}

/// One IOPATH delay; launch_edge is "posedge" or "negedge" for a clock-to-output arc.
#[pyclass(name = "IopathDelay", get_all, frozen)]
pub struct PyIopathDelay {
    pub from_port: String,
    pub to_port: String,
    pub launch_edge: Option<&'static str>,
    pub early_ps: DelayT,
    pub late_ps: DelayT,
}

/// Setup and hold limits of a data pin against a clock edge ("posedge" or "negedge").
#[pyclass(name = "TimingCheck", get_all, frozen)]
pub struct PyTimingCheck {
    pub data_port: String,
    pub clock_port: String,
    pub clock_edge: &'static str,
    pub setup_ps: DelayT,
    pub hold_ps: DelayT,
}

/// The IOPATH delays and timing checks of one cell instance.
#[pyclass(name = "CellDelays", get_all, frozen)]
pub struct PyCellDelays {
    pub instance: String,
    pub cell_type: String,
    pub iopaths: Vec<Py<PyIopathDelay>>,
    pub checks: Vec<Py<PyTimingCheck>>,
}

/// Routed delays of the top module `design`, cells and interconnect sorted by name.
#[pyclass(name = "BackAnnotation", get_all, frozen)]
pub struct PyBackAnnotation {
    pub design: String,
    pub cells: Vec<Py<PyCellDelays>>,
    pub interconnects: Vec<Py<PyInterconnectDelay>>,
}

/// One FASM feature of `tile`; `bits` is None for a set feature, else its value LSB first.
#[pyclass(name = "FasmFeature", get_all, frozen)]
pub struct PyFasmFeature {
    pub tile: String,
    pub feature: String,
    pub bits: Option<Vec<bool>>,
}

impl From<fasm::FasmFeature> for PyFasmFeature {
    fn from(feature: fasm::FasmFeature) -> Self {
        PyFasmFeature {
            tile: feature.tile,
            feature: feature.feature,
            bits: match feature.setting {
                fasm::FeatureSetting::Enabled => None,
                fasm::FeatureSetting::Bits { lsb_first } => Some(lsb_first),
            },
        }
    }
}

fn record<T: pyo3::PyClass>(py: Python<'_>, value: T) -> Py<T>
where
    T: Into<pyo3::PyClassInitializer<T>>,
{
    Py::new(py, value).expect("allocating a frozen record cannot fail")
}

fn sdf_edge(edge: ClockEdge) -> &'static str {
    match edge {
        ClockEdge::Rising => "posedge",
        ClockEdge::Falling => "negedge",
    }
}

impl From<BackAnnotation> for PyBackAnnotation {
    fn from(annotation: BackAnnotation) -> Self {
        Python::attach(|py| {
            let bounds = |delay: DelayBounds| (delay.early_ps, delay.late_ps);
            PyBackAnnotation {
                design: annotation.design,
                cells: annotation
                    .cells
                    .into_iter()
                    .map(|cell| {
                        record(py, PyCellDelays {
                            instance: cell.instance,
                            cell_type: cell.cell_type,
                            iopaths: cell
                                .iopaths
                                .into_iter()
                                .map(|arc| {
                                    let (early_ps, late_ps) = bounds(arc.delay);
                                    record(py, PyIopathDelay {
                                        from_port: arc.from,
                                        to_port: arc.to,
                                        launch_edge: arc.launch_edge.map(sdf_edge),
                                        early_ps,
                                        late_ps,
                                    })
                                })
                                .collect(),
                            checks: cell
                                .checks
                                .into_iter()
                                .map(|check| {
                                    record(py, PyTimingCheck {
                                        data_port: check.data,
                                        clock_port: check.clock,
                                        clock_edge: sdf_edge(check.edge),
                                        setup_ps: check.setup_ps,
                                        hold_ps: check.hold_ps,
                                    })
                                })
                                .collect(),
                        })
                    })
                    .collect(),
                interconnects: annotation
                    .interconnects
                    .into_iter()
                    .map(|net| {
                        let (early_ps, late_ps) = bounds(net.delay);
                        record(py, PyInterconnectDelay {
                            source_instance: net.source.instance,
                            source_port: net.source.port,
                            load_instance: net.load.instance,
                            load_port: net.load.port,
                            early_ps,
                            late_ps,
                        })
                    })
                    .collect(),
            }
        })
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

/// Run the upstream Himbächel flow using a trusted bridge library and chipdb.
#[pyfunction(signature = (*, library, device, chipdb, input, output, placer=None, router=None, seed=1, stop_after="route"))]
#[allow(clippy::too_many_arguments)]
fn run_upstream(
    py: Python<'_>,
    library: PathBuf,
    device: &str,
    chipdb: PathBuf,
    input: PathBuf,
    output: PathBuf,
    placer: Option<&str>,
    router: Option<&str>,
    seed: u64,
    stop_after: &str,
) -> PyResult<()> {
    let placer = match placer {
        Some("heap") => Some(UpstreamPlacer::Heap),
        Some("sa") => Some(UpstreamPlacer::Sa),
        Some(other) => {
            return Err(PyValueError::new_err(format!(
                "Unknown upstream placer: {other}"
            )))
        }
        None => None,
    };
    let router = match router {
        Some("router1") => Some(UpstreamRouter::Router1),
        Some("router2") => Some(UpstreamRouter::Router2),
        Some(other) => {
            return Err(PyValueError::new_err(format!(
                "Unknown upstream router: {other}"
            )))
        }
        None => None,
    };
    let stop_after = match stop_after {
        "pack" => Stage::Pack,
        "place" => Stage::Place,
        "route" => Stage::Route,
        other => {
            return Err(PyValueError::new_err(format!(
                "Unknown upstream stage: {other}"
            )))
        }
    };
    let request = Request {
        device,
        chipdb: &chipdb,
        input: &input,
        output: &output,
        placer,
        router,
        seed,
        stop_after,
    };
    py.detach(|| unsafe { Nextpnr::load(&library)?.run(&request) })
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

/// The `nextpnr` Python module.
#[pymodule]
fn nextpnr(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyContext>()?;
    m.add("RustContext", m.getattr("Context")?)?;
    m.add_class::<PyTimingReport>()?;
    m.add_class::<PyUtilizationReport>()?;
    m.add_class::<PyBackAnnotation>()?;
    m.add_class::<PyCellDelays>()?;
    m.add_class::<PyIopathDelay>()?;
    m.add_class::<PyTimingCheck>()?;
    m.add_class::<PyInterconnectDelay>()?;
    m.add_class::<PyFasmFeature>()?;
    m.add_function(wrap_pyfunction!(run_upstream, m)?)?;
    Ok(())
}
