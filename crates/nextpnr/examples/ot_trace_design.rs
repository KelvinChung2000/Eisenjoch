//! Generic `opt_trans` trace driver: run the placer on any (chipdb, design)
//! pair and let the per-iter `E_decomp` lines out on stderr.
//!
//! Unlike `ot_trace_stereovision3` nothing is hardcoded — both paths are
//! required env vars and a missing one is a hard error, not a default.
//! FPGA01 in particular lives only in the main checkout: `benchmark/**/*.json`
//! is gitignored, so it never materialises inside a worktree.
//!
//!   NPNR_OT_TRACE_CHIPDB=chip_database/xc7_large.bin \
//!   NPNR_OT_TRACE_DESIGN=benchmark/ispd/generated/2016/FPGA01/FPGA01.json \
//!   NPNR_OT_MAX_ITERS=20 cargo run --release --example ot_trace_design
//!
//! Congestion arms are selected purely through the placer's own env knobs:
//!   NPNR_OT_BPR_ALPHA=0     disable both BPR channels (wire + switch matrix)
//!   NPNR_OT_HARDEN_STEP=x   enable the non-decaying PathFinder history term

use nextpnr::chipdb::ChipDb;
use nextpnr::context::Context;
use nextpnr::frontend::parse_json;
use nextpnr::packer;
use nextpnr::placer::opt_trans::{OptTransPlacerCfg, PlacerOptTrans};
use nextpnr::placer::Placer;
use std::env;
use std::path::Path;

fn required(var: &str) -> String {
    env::var(var).unwrap_or_else(|_| panic!("{var} must be set"))
}

/// Background RSS sampler. Prints VmRSS/VmHWM every 250ms on stderr so the
/// growth curve interleaves with the placer's own phase markers. Diagnostic
/// only -- this driver is a measurement vehicle, not a shipped path.
fn spawn_rss_sampler() {
    std::thread::spawn(|| {
        let t0 = std::time::Instant::now();
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
                            .and_then(|v| v.parse::<usize>().ok())
                            .expect("parse /proc field");
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

fn main() {
    spawn_rss_sampler();
    let chipdb = required("NPNR_OT_TRACE_CHIPDB");
    let design = required("NPNR_OT_TRACE_DESIGN");
    eprintln!("chipdb: {chipdb}");
    eprintln!("design: {design}");

    let db = ChipDb::load(Path::new(&chipdb)).expect("load chipdb");
    let mut ctx = Context::new(db);
    let json = std::fs::read_to_string(&design).expect("read design");
    ctx.design = parse_json(&json, &ctx.id_pool).expect("parse design");
    packer::pack(&mut ctx, None).expect("pack");

    let mut cfg = OptTransPlacerCfg::default();
    cfg.seed = 42;
    cfg.report_interval = 1;
    // Everything else — iteration count, hardening step, BPR alpha, sweep
    // mode — comes from the placer's own env overrides so an arm is exactly
    // the set of vars named on the command line.
    cfg.apply_env_overrides();
    eprintln!(
        "cfg: max_outer_iters={} blend_alpha={} hardening_step={} graph_model={:?}",
        cfg.max_outer_iters, cfg.blend_alpha, cfg.hardening_step, cfg.graph_model,
    );

    PlacerOptTrans.place(&mut ctx, &cfg).expect("place");
}
