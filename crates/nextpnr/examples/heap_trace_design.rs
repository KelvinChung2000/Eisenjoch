//! HeAP counterpart to `ot_trace_design`: same chipdb, same design, same load
//! path, so placer wall-clock and peak RSS are directly comparable.
//!
//! It exists to answer "how long *should* this design take?" — `opt_trans`
//! being slow is only meaningful against the algorithm it is standing in for.
//! Both placers default to 20 outer iterations, so the comparison is at equal
//! iteration counts.
//!
//!   NPNR_HEAP_TRACE_CHIPDB=chip_database/xc7_large.bin \
//!   NPNR_HEAP_TRACE_DESIGN=benchmark/ispd/generated/2016/FPGA01/FPGA01.json \
//!   cargo run --release --example heap_trace_design
//!
//!   NPNR_HEAP_MAX_ITERS=n   override the outer iteration count

use nextpnr::chipdb::ChipDb;
use nextpnr::context::Context;
use nextpnr::frontend::parse_json;
use nextpnr::packer;
use nextpnr::placer::heap::{PlacerHeap, PlacerHeapCfg};
use nextpnr::metrics::{total_hpwl, total_line_estimate};
use nextpnr::placer::place_common::{get_net_metric, MetricType, WirelenT};
use nextpnr::placer::Placer;
use std::env;
use std::path::Path;
use std::time::Instant;

fn required(var: &str) -> String {
    env::var(var).unwrap_or_else(|_| panic!("{var} must be set"))
}

/// Background RSS sampler, byte-for-byte the behaviour of `ot_trace_design`'s
/// so the two logs can be compared with the same greps.
fn spawn_rss_sampler() {
    std::thread::spawn(|| {
        let t0 = Instant::now();
        loop {
            let s = std::fs::read_to_string("/proc/self/status")
                .expect("read /proc/self/status");
            let field = |name: &str| -> usize {
                for line in s.lines() {
                    if let Some(rest) = line.strip_prefix(name) {
                        return rest
                            .trim()
                            .split_whitespace()
                            .next()
                            .and_then(|v| v.parse().ok())
                            .unwrap_or(0);
                    }
                }
                0
            };
            eprintln!(
                "RSS_SAMPLE t={:.1}s rss={:.0}MB hwm={:.0}MB",
                t0.elapsed().as_secs_f64(),
                field("VmRSS:") as f64 / 1024.0,
                field("VmHWM:") as f64 / 1024.0,
            );
            std::thread::sleep(std::time::Duration::from_millis(250));
        }
    });
}

/// Sum of `get_net_metric(WIRELENGTH)` over every net — the same referee
/// `npnr_baseline_placers` uses, validated against upstream's own function.
fn score(ctx: &Context) -> (WirelenT, usize) {
    let mut total: WirelenT = 0;
    let mut nets = 0usize;
    let mut tns = 0.0;
    for net_idx in ctx.design.iter_net_indices() {
        let net = ctx.net(net_idx);
        if !net.is_alive() {
            continue;
        }
        nets += 1;
        total += get_net_metric(ctx, net.id(), MetricType::Wirelength, &mut tns);
    }
    (total, nets)
}

fn main() {
    spawn_rss_sampler();
    let chipdb = required("NPNR_HEAP_TRACE_CHIPDB");
    let design = required("NPNR_HEAP_TRACE_DESIGN");
    eprintln!("chipdb: {chipdb}");
    eprintln!("design: {design}");

    let t_load = Instant::now();
    let db = ChipDb::load(Path::new(&chipdb)).expect("load chipdb");
    let mut ctx = Context::new(db);
    let json = std::fs::read_to_string(&design).expect("read design");
    ctx.design = parse_json(&json, &ctx.id_pool).expect("parse design");
    packer::pack(&mut ctx, None).expect("pack");
    eprintln!("load+pack: {:.1}s", t_load.elapsed().as_secs_f64());

    let mut cfg = PlacerHeapCfg::default();
    cfg.seed = 42;
    if let Ok(v) = env::var("NPNR_HEAP_MAX_ITERS") {
        cfg.max_iterations = v.parse().expect("NPNR_HEAP_MAX_ITERS must be an integer");
    }
    eprintln!("cfg: max_iterations={}", cfg.max_iterations);

    let t_place = Instant::now();
    PlacerHeap.place(&mut ctx, &cfg).expect("place");
    let place_secs = t_place.elapsed().as_secs_f64();

    // Two metrics on purpose. `total_hpwl` is what opt_trans's own
    // post-legalization line reports, so it is the only one comparable against
    // `ot_trace_design`'s output; `get_net_metric` is the referee validated
    // against upstream nextpnr. Printing both keeps the comparison honest and
    // shows how far apart the two definitions actually are.
    // Guard the comparison: an unplaced or origin-stacked cell would make the
    // HPWL look *better*, not worse, so the count has to be checked before any
    // quality claim is made from it.
    let mut placed = 0usize;
    let mut unplaced = 0usize;
    for (c, _) in ctx.design.iter_alive_cells() {
        if ctx.cell(c).bel().is_some() { placed += 1 } else { unplaced += 1 }
    }
    eprintln!("cells: placed={placed} unplaced={unplaced}");
    assert_eq!(unplaced, 0, "HeAP left {unplaced} cells unplaced");

    let (wl, nets) = score(&ctx);
    let hpwl = total_hpwl(&ctx);
    let line = total_line_estimate(&ctx);
    eprintln!("HeAP place: {place_secs:.1}s");
    eprintln!("Post-placement: HPWL={hpwl:.0}, line={line:.0}");
    eprintln!("HeAP get_net_metric: {wl} over {nets} nets");
    println!(
        "HEAP_RESULT place_secs={place_secs:.1} total_hpwl={hpwl:.0} line={line:.0} \
get_net_metric={wl} nets={nets}"
    );
}
