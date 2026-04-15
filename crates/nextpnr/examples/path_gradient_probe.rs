use nextpnr::chipdb::ChipDb;
use nextpnr::context::Context;
use nextpnr::frontend::parse_json;
use nextpnr::packer;
use nextpnr::placer::PlacerPipeline;
use rustc_hash::FxHashSet;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::error::Error;
use std::path::Path;
use std::time::{Duration, Instant};

const CHIPDB_PATH: &str = "/home/kelvin/side-project/eisenjoch/chip_database/xc7_hybrid.bin";
const DESIGN_PATH: &str = "/home/kelvin/side-project/eisenjoch/benchmark/output/stereovision3.json";
const DELTAS: [f64; 4] = [1e-5, 1e-4, 1e-3, 1e-2];
const FD_TARGET_DELTA: f64 = 1e-4;
const SIGN_FLIP_DELTA: f64 = 1e-3;
const GRADIENT_CAP: f64 = 1e6;
const CUT_SAMPLES: usize = 50;
const DIJKSTRA_BENCH_RUNS: usize = 10;
const ESTIMATED_TOTAL_NETS: usize = 300;
const CG_BASELINE_SECONDS: f64 = 178.0;

pub mod chipdb {
    pub use nextpnr::chipdb::*;
}

pub mod context {
    pub use nextpnr::context::*;
}

pub mod netlist {
    pub use nextpnr::netlist::*;
}

pub mod metrics {
    pub mod congestion {
        pub use nextpnr::metrics::congestion::*;
    }
}

#[macro_export]
macro_rules! read_packed {
    ($base:expr, $field:ident) => {
        std::ptr::read_unaligned(std::ptr::addr_of!((*std::ptr::addr_of!($base)).$field))
    };
}

mod phase0_opt_trans {
    pub mod config {
        pub use nextpnr::placer::opt_trans::config::*;
    }

    #[path = "/home/kelvin/side-project/eisenjoch/crates/nextpnr/src/placer/opt_trans/network.rs"]
    pub mod network;

    #[path = "/home/kelvin/side-project/eisenjoch/crates/nextpnr/src/placer/opt_trans/demand.rs"]
    pub mod demand;
}

use phase0_opt_trans::demand::{collect_nets_for_solve, NetPinData, NetSolveInfo};
use phase0_opt_trans::network::PipeNetwork;

#[derive(Clone, Copy, Debug, PartialEq)]
struct HeapEntry {
    dist: f64,
    node: usize,
}

impl Eq for HeapEntry {}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .dist
            .total_cmp(&self.dist)
            .then_with(|| other.node.cmp(&self.node))
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Debug)]
struct NetEval {
    dist: Vec<f64>,
    cost: f64,
    source_node: usize,
}

#[derive(Clone, Debug)]
struct CutStats {
    sink_pin_idx: usize,
    samples: usize,
    max_jump: f64,
    max_bound: f64,
    violations: usize,
}

#[derive(Clone, Debug)]
struct DijkstraBench {
    runs: usize,
    mean: Duration,
    p99: Duration,
}

#[derive(Default, Debug)]
struct ProbeSummary {
    probed_nets: usize,
    probed_pins: usize,
    fd_checked_pins_at_target: usize,
    fd_pass_pins_at_target: usize,
    worst_fd_error: f64,
    max_analytical_grad: f64,
    sign_flip_count: usize,
    cut_violation_count: usize,
    gradient_blowup_count: usize,
    invalid_pin_count: usize,
    sign_flip_details: Vec<String>,
    cut_violation_details: Vec<String>,
    blowup_details: Vec<String>,
    invalid_pin_details: Vec<String>,
    net_observations: Vec<String>,
    net_timing_details: Vec<String>,
    total_baseline_eval: Duration,
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("path_gradient_probe: loading stereovision3");

    let db = ChipDb::load(Path::new(CHIPDB_PATH))?;
    let mut ctx = Context::new(db);
    let json = std::fs::read_to_string(DESIGN_PATH)?;
    ctx.design = parse_json(&json, &ctx.id_pool)?;
    packer::pack(&mut ctx, None)?;

    let setup = PlacerPipeline::prepare(&mut ctx, 42)?;

    let level0_scale = 2.0;
    let default_cfg = nextpnr::placer::opt_trans::OptTransPlacerCfg::default();
    let network = PipeNetwork::from_context(&ctx, level0_scale, &default_cfg);

    let coarse_x: Vec<f64> = setup
        .cell_x
        .iter()
        .map(|&x| x / network.coarsen as f64)
        .collect();
    let coarse_y: Vec<f64> = setup
        .cell_y
        .iter()
        .map(|&y| y / network.coarsen as f64)
        .collect();
    let alive_net_ids: Vec<_> = ctx
        .design
        .iter_alive_nets()
        .map(|(net_id, _)| net_id)
        .collect();
    let net_infos = collect_nets_for_solve(
        &ctx,
        &alive_net_ids,
        true,
        &setup.cell_to_idx,
        &coarse_x,
        &coarse_y,
        &network,
    );

    println!(
        "production scale={:.3} coarsen={} grid={}x{} nodes={} pipes={} nets={}",
        level0_scale,
        network.coarsen,
        network.width,
        network.height,
        network.num_nodes(),
        network.num_pipes(),
        net_infos.len(),
    );

    let probe_net_indices = select_probe_nets(&net_infos);
    if probe_net_indices.is_empty() {
        return Err("no probe nets with movable sinks found".into());
    }

    println!(
        "selected {} probe nets: {}",
        probe_net_indices.len(),
        probe_net_indices
            .iter()
            .map(|&idx| format!(
                "{}({} pins)",
                net_label(&net_infos[idx]),
                net_infos[idx].pins.len()
            ))
            .collect::<Vec<_>>()
            .join(", "),
    );

    let mut summary = ProbeSummary {
        probed_nets: probe_net_indices.len(),
        ..ProbeSummary::default()
    };
    let dijkstra_bench = benchmark_dijkstra(&network);

    for &net_idx in &probe_net_indices {
        probe_net(&network, &net_infos[net_idx], &mut summary);
    }

    let fd_pass_rate = if summary.fd_checked_pins_at_target > 0 {
        summary.fd_pass_pins_at_target as f64 / summary.fd_checked_pins_at_target as f64
    } else {
        0.0
    };
    let fd_pass = fd_pass_rate >= 0.80;
    let sign_flip_pass = summary.sign_flip_count == 0;
    let cut_pass = summary.cut_violation_count == 0;
    let gradient_pass =
        summary.max_analytical_grad < GRADIENT_CAP && summary.gradient_blowup_count == 0;
    let rung0_pass = fd_pass && sign_flip_pass && cut_pass && gradient_pass;

    println!();
    println!("=== PATH GRADIENT PROBE SUMMARY ===");
    println!("probed_nets: {}", summary.probed_nets);
    println!("probed_pins: {}", summary.probed_pins);
    println!(
        "rung_0_fd_check: {}/{} pins within 5% at delta={:.0e} => {}",
        summary.fd_pass_pins_at_target,
        summary.fd_checked_pins_at_target,
        FD_TARGET_DELTA,
        verdict(fd_pass),
    );
    println!(
        "rung_0_sign_flip_check: {} detected => {}",
        summary.sign_flip_count,
        verdict(sign_flip_pass),
    );
    println!(
        "rung_0_cut_continuity: {} violations => {}",
        summary.cut_violation_count,
        verdict(cut_pass),
    );
    println!(
        "rung_0_gradient_bound: max |grad|={:.6e} cap={:.1e} blowups={} => {}",
        summary.max_analytical_grad,
        GRADIENT_CAP,
        summary.gradient_blowup_count,
        verdict(gradient_pass),
    );
    println!(
        "worst_fd_vs_analytical_error: {:.6e}",
        summary.worst_fd_error
    );
    println!("invalid_pins_skipped: {}", summary.invalid_pin_count);

    if !summary.sign_flip_details.is_empty() {
        println!("sign_flip_details:");
        for detail in summary.sign_flip_details.iter().take(8) {
            println!("  {detail}");
        }
    }
    if !summary.cut_violation_details.is_empty() {
        println!("cut_violation_details:");
        for detail in summary.cut_violation_details.iter().take(8) {
            println!("  {detail}");
        }
    }
    if !summary.blowup_details.is_empty() {
        println!("gradient_blowup_details:");
        for detail in summary.blowup_details.iter().take(8) {
            println!("  {detail}");
        }
    }
    if !summary.invalid_pin_details.is_empty() {
        println!("invalid_pin_details:");
        for detail in summary.invalid_pin_details.iter().take(8) {
            println!("  {detail}");
        }
    }

    println!("observations:");
    for observation in &summary.net_observations {
        println!("  {observation}");
    }
    println!("timing:");
    println!(
        "  single_dijkstra_{}runs: mean={} p99={}",
        dijkstra_bench.runs,
        fmt_duration(dijkstra_bench.mean),
        fmt_duration(dijkstra_bench.p99),
    );
    for detail in &summary.net_timing_details {
        println!("  {detail}");
    }
    let estimated_gradient = scale_duration(
        summary.total_baseline_eval,
        ESTIMATED_TOTAL_NETS as f64 / summary.probed_nets as f64,
    );
    let speedup_vs_cg = CG_BASELINE_SECONDS / estimated_gradient.as_secs_f64().max(1e-12);
    println!(
        "  total_probe_net_eval_time: {} for {} nets",
        fmt_duration(summary.total_baseline_eval),
        summary.probed_nets,
    );
    println!(
        "  estimated_gradient_cost_{}nets: {} ({:.2}x vs {:.1}s CG baseline)",
        ESTIMATED_TOTAL_NETS,
        fmt_duration(estimated_gradient),
        speedup_vs_cg,
        CG_BASELINE_SECONDS,
    );
    println!(
        "{}",
        if rung0_pass {
            "RUNG 0 PASSED"
        } else {
            "RUNG 0 FAILED, escalate to Rung 1"
        }
    );

    Ok(())
}

fn verdict(pass: bool) -> &'static str {
    if pass {
        "PASS"
    } else {
        "FAIL"
    }
}

fn net_label(info: &NetSolveInfo) -> String {
    if info.debug_name.is_empty() {
        format!("{:?}", info.net_id)
    } else {
        info.debug_name.clone()
    }
}

fn movable_sink_indices(info: &NetSolveInfo) -> Vec<usize> {
    info.pin_data
        .iter()
        .enumerate()
        .filter_map(|(idx, pin)| {
            if pin.is_driver || pin.is_fixed || pin.cell_idx.is_none() {
                None
            } else {
                Some(idx)
            }
        })
        .collect()
}

fn select_probe_nets(net_infos: &[NetSolveInfo]) -> Vec<usize> {
    let mut selected = Vec::new();
    let mut used = FxHashSet::default();

    let bins = [
        (3usize, 2usize, 2usize),
        (3usize, 3usize, 5usize),
        (3usize, 10usize, 20usize),
    ];

    for (target, lo, hi) in bins {
        let mut taken = 0usize;
        for (idx, info) in net_infos.iter().enumerate() {
            if used.contains(&idx) || !has_movable_sink(info) {
                continue;
            }
            let pins = info.pins.len();
            if pins < lo || pins > hi {
                continue;
            }
            selected.push(idx);
            used.insert(idx);
            taken += 1;
            if taken >= target {
                break;
            }
        }
    }

    for (idx, info) in net_infos.iter().enumerate() {
        if used.contains(&idx) || !has_movable_sink(info) {
            continue;
        }
        if info.pins.len() > 20 {
            selected.push(idx);
            used.insert(idx);
            break;
        }
    }

    for (idx, info) in net_infos.iter().enumerate() {
        if selected.len() >= 10 {
            break;
        }
        if used.contains(&idx) || !has_movable_sink(info) {
            continue;
        }
        selected.push(idx);
        used.insert(idx);
    }

    selected
}

fn has_movable_sink(info: &NetSolveInfo) -> bool {
    info.pin_data
        .iter()
        .any(|pin| !pin.is_driver && !pin.is_fixed && pin.cell_idx.is_some())
}

fn probe_net(network: &PipeNetwork, base_info: &NetSolveInfo, summary: &mut ProbeSummary) {
    let label = net_label(base_info);
    let movable_sinks = movable_sink_indices(base_info);
    println!();
    println!(
        "-- net {}: total_pins={} movable_sinks={}",
        label,
        base_info.pins.len(),
        movable_sinks.len()
    );

    let baseline_start = Instant::now();
    let Some(base_eval) = evaluate_net_cost(base_info, network) else {
        let detail = format!("{label}: baseline net cost invalid or disconnected");
        println!("  skipped: {detail}");
        summary.invalid_pin_count += movable_sinks.len();
        summary.invalid_pin_details.push(detail);
        return;
    };
    let baseline_elapsed = baseline_start.elapsed();
    summary.total_baseline_eval += baseline_elapsed;
    summary.net_timing_details.push(format!(
        "{}: baseline_eval={} cost={:.6e} source={}",
        label,
        fmt_duration(baseline_elapsed),
        base_eval.cost,
        base_eval.source_node,
    ));

    println!(
        "  source_node={} baseline_cost={:.6e} baseline_eval={}",
        base_eval.source_node,
        base_eval.cost,
        fmt_duration(baseline_elapsed),
    );

    let mut per_net_cut: Option<CutStats> = None;
    let mut worst_pin_err = 0.0f64;
    let mut net_max_grad = 0.0f64;
    let mut net_sign_flips = 0usize;

    for (movable_ord, &pin_idx) in movable_sinks.iter().enumerate() {
        let pin = &base_info.pin_data[pin_idx];
        let Some((analytical_dx, analytical_dy)) = sink_gradient(pin, &base_eval.dist) else {
            let detail = format!("{label}: pin {pin_idx} analytical gradient invalid");
            println!("  skipped pin {pin_idx}: invalid analytical gradient");
            summary.invalid_pin_count += 1;
            summary.invalid_pin_details.push(detail);
            continue;
        };

        summary.probed_pins += 1;
        let pin_grad_max = analytical_dx.abs().max(analytical_dy.abs());
        summary.max_analytical_grad = summary.max_analytical_grad.max(pin_grad_max);
        net_max_grad = net_max_grad.max(pin_grad_max);
        if pin_grad_max >= GRADIENT_CAP {
            let detail = format!(
                "{label}: pin {pin_idx} |grad| component max {:.6e} exceeds cap",
                pin_grad_max
            );
            summary.gradient_blowup_count += 1;
            summary.blowup_details.push(detail);
        }

        let mut delta_1e4_pass = false;
        println!(
            "  pin {}: analytical dx={:.6e} dy={:.6e}",
            pin_idx, analytical_dx, analytical_dy
        );

        for &delta in &DELTAS {
            let x_plus = shifted_net_info(base_info, network, pin_idx, delta, 0.0);
            let x_minus = shifted_net_info(base_info, network, pin_idx, -delta, 0.0);
            let y_plus = shifted_net_info(base_info, network, pin_idx, 0.0, delta);
            let y_minus = shifted_net_info(base_info, network, pin_idx, 0.0, -delta);

            let (Some(x_plus_eval), Some(x_minus_eval), Some(y_plus_eval), Some(y_minus_eval)) = (
                evaluate_net_cost(&x_plus, network),
                evaluate_net_cost(&x_minus, network),
                evaluate_net_cost(&y_plus, network),
                evaluate_net_cost(&y_minus, network),
            ) else {
                let detail =
                    format!("{label}: pin {pin_idx} invalid FD evaluation at delta={delta:.0e}");
                println!("    delta={delta:.0e}: invalid finite-difference evaluation");
                summary.invalid_pin_count += 1;
                summary.invalid_pin_details.push(detail);
                continue;
            };

            let fd_dx = (x_plus_eval.cost - x_minus_eval.cost) / (2.0 * delta);
            let fd_dy = (y_plus_eval.cost - y_minus_eval.cost) / (2.0 * delta);
            let rel_err_dx = rel_error(fd_dx, analytical_dx);
            let rel_err_dy = rel_error(fd_dy, analytical_dy);
            let pin_worst = rel_err_dx.max(rel_err_dy);
            summary.worst_fd_error = summary.worst_fd_error.max(pin_worst);
            worst_pin_err = worst_pin_err.max(pin_worst);

            println!(
                "    delta={delta:.0e}: fd_dx={:.6e} err_x={:.3e} fd_dy={:.6e} err_y={:.3e}",
                fd_dx, rel_err_dx, fd_dy, rel_err_dy
            );

            if (delta - FD_TARGET_DELTA).abs() <= f64::EPSILON {
                summary.fd_checked_pins_at_target += 1;
                if pin_worst <= 0.05 {
                    summary.fd_pass_pins_at_target += 1;
                    delta_1e4_pass = true;
                }
            }
        }

        let pin_sign_flips =
            detect_sign_flips(base_info, network, pin_idx, &base_eval.dist, &label);
        net_sign_flips += pin_sign_flips.len();
        summary.sign_flip_count += pin_sign_flips.len();
        summary.sign_flip_details.extend(pin_sign_flips);

        if movable_ord == 0 {
            per_net_cut = Some(run_1d_cut(base_info, network, pin_idx, summary));
        }

        println!(
            "    delta={:.0e} status: {}",
            FD_TARGET_DELTA,
            verdict(delta_1e4_pass),
        );
    }

    let cut_note = if let Some(cut) = per_net_cut {
        format!(
            "cut pin={} samples={} max_jump={:.3e} bound={:.3e} violations={}",
            cut.sink_pin_idx, cut.samples, cut.max_jump, cut.max_bound, cut.violations
        )
    } else {
        "cut unavailable".to_string()
    };

    let topology_note = if net_sign_flips > 0 || worst_pin_err > 0.05 {
        "topology-driven kinks visible"
    } else {
        "mostly Manhattan-like / bilinear-smooth at production resolution"
    };

    let observation = format!(
        "{}: source={} max|grad|={:.3e} worst_fd_err={:.3e} sign_flips={} {} {}",
        label,
        base_eval.source_node,
        net_max_grad,
        worst_pin_err,
        net_sign_flips,
        cut_note,
        topology_note,
    );
    println!("  observation: {observation}");
    summary.net_observations.push(observation);
}

fn detect_sign_flips(
    base_info: &NetSolveInfo,
    network: &PipeNetwork,
    pin_idx: usize,
    _base_dist: &[f64],
    label: &str,
) -> Vec<String> {
    let mut details = Vec::new();

    let x_plus = shifted_net_info(base_info, network, pin_idx, SIGN_FLIP_DELTA, 0.0);
    let x_minus = shifted_net_info(base_info, network, pin_idx, -SIGN_FLIP_DELTA, 0.0);
    if x_plus.pin_data[pin_idx].nodes == x_minus.pin_data[pin_idx].nodes {
        if let (Some(xp), Some(xm)) = (
            evaluate_net_cost(&x_plus, network),
            evaluate_net_cost(&x_minus, network),
        ) {
            if let (Some((gpx, _)), Some((gmx, _))) = (
                sink_gradient(&x_plus.pin_data[pin_idx], &xp.dist),
                sink_gradient(&x_minus.pin_data[pin_idx], &xm.dist),
            ) {
                if signed(gpx) != 0 && signed(gpx) == -signed(gmx) {
                    details.push(format!(
                        "{label}: pin {pin_idx} dx sign flip across ±{SIGN_FLIP_DELTA:.0e} within one cell ({gmx:.3e} -> {gpx:.3e})"
                    ));
                }
            }
        }
    }

    let y_plus = shifted_net_info(base_info, network, pin_idx, 0.0, SIGN_FLIP_DELTA);
    let y_minus = shifted_net_info(base_info, network, pin_idx, 0.0, -SIGN_FLIP_DELTA);
    if y_plus.pin_data[pin_idx].nodes == y_minus.pin_data[pin_idx].nodes {
        if let (Some(yp), Some(ym)) = (
            evaluate_net_cost(&y_plus, network),
            evaluate_net_cost(&y_minus, network),
        ) {
            if let (Some((_, gpy)), Some((_, gmy))) = (
                sink_gradient(&y_plus.pin_data[pin_idx], &yp.dist),
                sink_gradient(&y_minus.pin_data[pin_idx], &ym.dist),
            ) {
                if signed(gpy) != 0 && signed(gpy) == -signed(gmy) {
                    details.push(format!(
                        "{label}: pin {pin_idx} dy sign flip across ±{SIGN_FLIP_DELTA:.0e} within one cell ({gmy:.3e} -> {gpy:.3e})"
                    ));
                }
            }
        }
    }

    details
}

fn run_1d_cut(
    base_info: &NetSolveInfo,
    network: &PipeNetwork,
    pin_idx: usize,
    summary: &mut ProbeSummary,
) -> CutStats {
    let (base_x, base_y, _) = base_info.pins[pin_idx];
    let half_span = 2.5;
    let start = base_x - half_span;
    let end = base_x + half_span;
    let step = if CUT_SAMPLES > 1 {
        (end - start) / (CUT_SAMPLES - 1) as f64
    } else {
        0.0
    };

    let mut samples = Vec::new();
    for sample_idx in 0..CUT_SAMPLES {
        let x = start + sample_idx as f64 * step;
        let info = replacement_net_info(base_info, network, pin_idx, x, base_y);
        if let Some(eval) = evaluate_net_cost(&info, network) {
            if let Some((grad_x, _)) = sink_gradient(&info.pin_data[pin_idx], &eval.dist) {
                samples.push((x, eval.cost, grad_x));
            }
        }
    }

    let mut stats = CutStats {
        sink_pin_idx: pin_idx,
        samples: samples.len(),
        max_jump: 0.0,
        max_bound: 0.0,
        violations: 0,
    };

    for pair in samples.windows(2) {
        let (x0, cost0, grad0) = pair[0];
        let (x1, cost1, grad1) = pair[1];
        let dx = (x1 - x0).abs();
        let jump = (cost1 - cost0).abs();
        let local_grad = grad0.abs().max(grad1.abs());
        let bound = 2.0 * local_grad * dx;
        stats.max_jump = stats.max_jump.max(jump);
        stats.max_bound = stats.max_bound.max(bound);
        if jump > bound + 1e-9 {
            stats.violations += 1;
            summary.cut_violation_count += 1;
            summary.cut_violation_details.push(format!(
                "{}: pin {} cut jump {:.3e} exceeds bound {:.3e} between x={:.3} and x={:.3}",
                net_label(base_info),
                pin_idx,
                jump,
                bound,
                x0,
                x1
            ));
        }
    }

    stats
}

fn shifted_net_info(
    base_info: &NetSolveInfo,
    network: &PipeNetwork,
    pin_idx: usize,
    dx: f64,
    dy: f64,
) -> NetSolveInfo {
    let (x, y, _) = base_info.pins[pin_idx];
    replacement_net_info(base_info, network, pin_idx, x + dx, y + dy)
}

fn replacement_net_info(
    base_info: &NetSolveInfo,
    network: &PipeNetwork,
    pin_idx: usize,
    new_x: f64,
    new_y: f64,
) -> NetSolveInfo {
    let mut pins = base_info.pins.clone();
    pins[pin_idx].0 = new_x;
    pins[pin_idx].1 = new_y;
    NetSolveInfo::from_pins(
        base_info.net_id,
        base_info.debug_name.clone(),
        pins,
        base_info.pin_is_fixed.clone(),
        base_info.has_fixed_pin,
        network,
    )
}

fn evaluate_net_cost(info: &NetSolveInfo, network: &PipeNetwork) -> Option<NetEval> {
    let source_node = nearest_source_node(info.pin_data.first()?)?;
    let dist = dijkstra_distances(network, source_node);
    let mut cost = 0.0;
    for pin in info.pin_data.iter().filter(|pin| !pin.is_driver) {
        cost += sink_cost(pin, &dist)?;
    }
    Some(NetEval {
        dist,
        cost,
        source_node,
    })
}

fn nearest_source_node(pin: &NetPinData) -> Option<usize> {
    let mut best_weight = f64::NEG_INFINITY;
    let mut best_node = None;
    for j in 0..4 {
        let weight = pin.weights[j];
        let node = pin.nodes[j];
        if weight > best_weight
            || ((weight - best_weight).abs() <= f64::EPSILON && Some(node) < best_node)
        {
            best_weight = weight;
            best_node = Some(node);
        }
    }
    best_node
}

fn dijkstra_distances(network: &PipeNetwork, source_node: usize) -> Vec<f64> {
    let mut dist = vec![f64::INFINITY; network.num_nodes()];
    let mut heap = BinaryHeap::new();

    dist[source_node] = 0.0;
    heap.push(HeapEntry {
        dist: 0.0,
        node: source_node,
    });

    while let Some(HeapEntry {
        dist: cur_dist,
        node,
    }) = heap.pop()
    {
        if cur_dist > dist[node] {
            continue;
        }

        for &pipe_idx in &network.node_pipes[node] {
            let pipe = &network.pipes[pipe_idx];
            let next = if pipe.from == node {
                pipe.to
            } else if pipe.to == node {
                pipe.from
            } else {
                continue;
            };

            if !pipe.eff_conductance.is_finite() || pipe.eff_conductance <= 0.0 {
                continue;
            }
            let edge_cost = 1.0 / pipe.eff_conductance;
            let candidate = cur_dist + edge_cost;
            if candidate < dist[next] {
                dist[next] = candidate;
                heap.push(HeapEntry {
                    dist: candidate,
                    node: next,
                });
            }
        }
    }

    dist
}

fn sink_cost(pin: &NetPinData, dist: &[f64]) -> Option<f64> {
    let mut cost = 0.0;
    for j in 0..4 {
        let weight = pin.weights[j];
        if weight == 0.0 {
            continue;
        }
        let d = dist[pin.nodes[j]];
        if !d.is_finite() {
            return None;
        }
        cost += weight * d;
    }
    Some(cost)
}

fn sink_gradient(pin: &NetPinData, dist: &[f64]) -> Option<(f64, f64)> {
    let mut grad_x = 0.0;
    let mut grad_y = 0.0;
    for j in 0..4 {
        let d = dist[pin.nodes[j]];
        if !d.is_finite() {
            return None;
        }
        grad_x += pin.dw_dx[j] * d;
        grad_y += pin.dw_dy[j] * d;
    }
    Some((grad_x, grad_y))
}

fn rel_error(fd: f64, analytical: f64) -> f64 {
    (fd - analytical).abs() / analytical.abs().max(1e-12)
}

fn signed(value: f64) -> i32 {
    if value > 1e-12 {
        1
    } else if value < -1e-12 {
        -1
    } else {
        0
    }
}

fn benchmark_dijkstra(network: &PipeNetwork) -> DijkstraBench {
    let mut samples = Vec::with_capacity(DIJKSTRA_BENCH_RUNS);
    let mut state = 0x9e37_79b9_7f4a_7c15u64;

    for _ in 0..DIJKSTRA_BENCH_RUNS {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let source = (state as usize) % network.num_nodes();
        let start = Instant::now();
        let _ = dijkstra_distances(network, source);
        samples.push(start.elapsed());
    }

    samples.sort_unstable();
    let total = samples
        .iter()
        .fold(Duration::ZERO, |acc, &sample| acc.saturating_add(sample));
    let mean = scale_duration(total, 1.0 / samples.len() as f64);
    let p99_index = ((samples.len() as f64) * 0.99).ceil() as usize - 1;
    let p99 = samples[p99_index.min(samples.len() - 1)];

    DijkstraBench {
        runs: DIJKSTRA_BENCH_RUNS,
        mean,
        p99,
    }
}

fn scale_duration(duration: Duration, factor: f64) -> Duration {
    Duration::from_secs_f64(duration.as_secs_f64() * factor)
}

fn fmt_duration(duration: Duration) -> String {
    if duration.as_secs_f64() >= 1.0 {
        format!("{:.3}s", duration.as_secs_f64())
    } else if duration.as_secs_f64() >= 1e-3 {
        format!("{:.3}ms", duration.as_secs_f64() * 1e3)
    } else {
        format!("{:.3}us", duration.as_secs_f64() * 1e6)
    }
}
