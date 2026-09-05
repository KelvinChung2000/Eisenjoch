//! Place FPGA01 with either placer, then route the result with the raster
//! router, so the two placements meet the same router under the same cap.
//!
//! Placement is fast enough now (~1.3s on FPGA01) that no checkpoint is
//! worth carrying: each arm places inline. The arm is one env var, and the
//! placer's own env knobs configure it, so an arm is exactly the set of
//! variables named on the command line.
//!
//!   NPNR_ROUTE_CHIPDB=chip_database/xc7_large.bin \
//!   NPNR_ROUTE_DESIGN=benchmark/ispd/generated/2016/FPGA01/FPGA01.json \
//!   NPNR_ROUTE_PLACER=opt_trans NPNR_ROUTE_ITERS=50 \
//!   cargo run --release --example route_after_place

use nextpnr::chipdb::ChipDb;
use nextpnr::context::Context;
use nextpnr::frontend::parse_json;
use nextpnr::packer;
use nextpnr::placer::heap::{PlacerHeap, PlacerHeapCfg};
use nextpnr::placer::opt_trans::{OptTransPlacerCfg, PlacerOptTrans};
use nextpnr::placer::Placer;
use nextpnr::router::common::collect_routable_nets;
use nextpnr::router::raster::{RasterRouter, RasterRouterCfg};
use nextpnr::router::Router;
use std::env;
use std::path::Path;
use std::time::Instant;

fn required(var: &str) -> String {
    env::var(var).unwrap_or_else(|_| panic!("{var} must be set"))
}

/// Nets carrying at least one bound wire. The per-pass router lines carry the
/// authoritative routed/failed split; this is the count that survives the cap.
fn nets_with_routing(ctx: &Context) -> (usize, usize) {
    let mut with = 0usize;
    let mut alive = 0usize;
    for net_idx in ctx.design.iter_net_indices() {
        let net = ctx.net(net_idx);
        if !net.is_alive() {
            continue;
        }
        alive += 1;
        if !net.wires().is_empty() {
            with += 1;
        }
    }
    (with, alive)
}

fn main() {
    let chipdb = required("NPNR_ROUTE_CHIPDB");
    let design = required("NPNR_ROUTE_DESIGN");
    let arm = required("NPNR_ROUTE_PLACER");
    let route_iters: usize = env::var("NPNR_ROUTE_ITERS")
        .unwrap_or_else(|_| "50".into())
        .parse()
        .expect("NPNR_ROUTE_ITERS must be an integer");
    eprintln!("chipdb: {chipdb}");
    eprintln!("design: {design}");
    eprintln!("placer: {arm}, route_iters: {route_iters}");

    let db = ChipDb::load(Path::new(&chipdb)).expect("load chipdb");
    let mut ctx = Context::new(db);
    let json = std::fs::read_to_string(&design).expect("read design");
    ctx.design = parse_json(&json, &ctx.id_pool).expect("parse design");
    packer::pack(&mut ctx, None).expect("pack");

    let t_place = Instant::now();
    match arm.as_str() {
        "opt_trans" => {
            let mut cfg = OptTransPlacerCfg::default();
            cfg.seed = 42;
            cfg.report_interval = 1;
            cfg.apply_env_overrides();
            PlacerOptTrans.place(&mut ctx, &cfg).expect("place");
        }
        "heap" => {
            let mut cfg = PlacerHeapCfg::default();
            cfg.seed = 42;
            if let Ok(v) = env::var("NPNR_HEAP_MAX_ITERS") {
                cfg.max_iterations = v.parse().expect("NPNR_HEAP_MAX_ITERS must be an integer");
            }
            PlacerHeap.place(&mut ctx, &cfg).expect("place");
        }
        other => panic!("NPNR_ROUTE_PLACER must be opt_trans or heap, got {other}"),
    }
    let place_secs = t_place.elapsed().as_secs_f64();
    let hpwl = nextpnr::metrics::wirelength::total_hpwl(&ctx);
    println!("PLACE_RESULT placer={arm} place_secs={place_secs:.2} total_hpwl={hpwl:.0}");

    let nets = collect_routable_nets(&ctx);
    eprintln!("routable nets: {}", nets.len());

    // A doomed sink spends ~214k pops under the kernel default, over half the
    // router's total. NPNR_ROUTE_SINK_VISITS caps that; unset keeps the default.
    let sink_visit_limit = env::var("NPNR_ROUTE_SINK_VISITS")
        .ok()
        .map(|v| v.parse().expect("NPNR_ROUTE_SINK_VISITS must be an integer"));
    eprintln!("sink_visit_limit: {sink_visit_limit:?}");

    let cleanup_visit_limit = env::var("NPNR_ROUTE_CLEANUP_VISITS")
        .ok()
        .map(|v| v.parse().expect("NPNR_ROUTE_CLEANUP_VISITS must be an integer"));
    eprintln!("cleanup_visit_limit: {cleanup_visit_limit:?}");

    // The finisher shared the negotiation cap until this knob split them, so
    // unset it inherits `sink_visit_limit` and every earlier measurement stays
    // comparable. `0` selects the kernel default, which scales with the sink's
    // distance from the tree instead of sitting flat.
    let finish_visit_limit = match env::var("NPNR_ROUTE_FINISH_VISITS").ok() {
        None => sink_visit_limit,
        Some(v) => match v.parse().expect("NPNR_ROUTE_FINISH_VISITS must be an integer") {
            0 => None,
            n => Some(n),
        },
    };
    eprintln!("finish_visit_limit: {finish_visit_limit:?}");

    // `0` removes the finisher's wall-clock per-net deadline. Raising the pop
    // budget raises per-sink cost by the same factor, so a fixed wall would
    // abandon the sinks the budget was raised to reach.
    let finish_net_deadline = match env::var("NPNR_ROUTE_FINISH_DEADLINE_MS").ok() {
        None => Some(std::time::Duration::from_millis(2000)),
        Some(v) => match v.parse().expect("NPNR_ROUTE_FINISH_DEADLINE_MS must be an integer") {
            0 => None,
            ms => Some(std::time::Duration::from_millis(ms)),
        },
    };
    eprintln!("finish_net_deadline: {finish_net_deadline:?}");

    // Cleanup recovers 1.4% of failed nets at 8.1s each. Skipping it asks
    // whether the pass loop's tile congestion history converges on its own.
    let skip_cleanup = env::var("NPNR_ROUTE_SKIP_CLEANUP").ok().as_deref() == Some("1");
    eprintln!("skip_cleanup: {skip_cleanup}");

    // Scoped rip-up in the pass loop; cleanup keeps only const/clock paths.
    let inloop_ripup = env::var("NPNR_ROUTE_INLOOP_RIPUP").ok().as_deref() == Some("1");
    eprintln!("inloop_ripup: {inloop_ripup}");

    // PathFinder over-use negotiation; NPNR_ROUTE_PRESENT / _PRESENT_GROWTH /
    // _HIST_GAIN tune the pricing schedule inside the router.
    let overuse = env::var("NPNR_ROUTE_OVERUSE").ok().as_deref() == Some("1");
    eprintln!("overuse: {overuse}");

    // Per-net bbox margin. The default 8 fences every search into a corridor
    // holding a fraction of the fabric's ~1957 wires per tile, while the
    // hottest tile on the chip uses ~700 of them: the residual demand fails
    // for want of reachable capacity, not existing capacity.
    let bbox_margin: i32 = env::var("NPNR_ROUTE_BBOX_MARGIN")
        .ok()
        .map(|v| v.parse().expect("NPNR_ROUTE_BBOX_MARGIN must be an integer"))
        .unwrap_or(8);
    eprintln!("bbox_margin: {bbox_margin}");

    let cfg = RasterRouterCfg {
        max_iterations: route_iters,
        verbose: true,
        sink_visit_limit,
        cleanup_visit_limit,
        finish_visit_limit,
        finish_net_deadline,
        skip_cleanup,
        inloop_ripup,
        overuse,
        bbox_margin,
        ..RasterRouterCfg::default()
    };
    let t_route = Instant::now();
    let outcome = RasterRouter.route_nets(&mut ctx, &cfg, &nets);
    let route_secs = t_route.elapsed().as_secs_f64();

    // An unrouted design is the measurement, not a crash: the error text is
    // printed and the trajectory reported rather than propagated as a panic.
    let (status, detail) = match &outcome {
        Ok(()) => ("complete".to_string(), String::new()),
        Err(e) => ("incomplete".to_string(), e.to_string()),
    };
    let (routed, alive) = nets_with_routing(&ctx);
    println!(
        "ROUTE_RESULT placer={arm} status={status} route_secs={route_secs:.1} \
nets_with_routing={routed} alive_nets={alive} routable={}",
        nets.len()
    );
    if !detail.is_empty() {
        eprintln!("ROUTE_ERROR {detail}");
    }
}
