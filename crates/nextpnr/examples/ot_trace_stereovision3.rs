use nextpnr::chipdb::ChipDb;
use nextpnr::context::Context;
use nextpnr::frontend::parse_json;
use nextpnr::packer;
use nextpnr::placer::opt_trans::{OptTransPlacerCfg, PlacerOptTrans};
use nextpnr::placer::Placer;
use std::env;
use std::path::Path;

fn main() {
    let chipdb = "/home/kelvin/side-project/eisenjoch/chip_database/xc7_hybrid.bin";
    let design = "/home/kelvin/side-project/eisenjoch/benchmark/output/stereovision3.json";

    let db = ChipDb::load(Path::new(chipdb)).expect("load chipdb");
    let mut ctx = Context::new(db);
    let json = std::fs::read_to_string(design).expect("read design");
    ctx.design = parse_json(&json, &ctx.id_pool).expect("parse design");
    packer::pack(&mut ctx, None).expect("pack");

    let mut cfg = OptTransPlacerCfg::default();
    cfg.seed = 42;
    cfg.max_iters = env::var("NPNR_OT_TRACE_MAX_ITERS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1);
    cfg.adam_lr_gain = env::var("NPNR_OT_TRACE_ADAM_LR_GAIN")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(cfg.adam_lr_gain);
    cfg.energy_progress_ema_beta = env::var("NPNR_OT_TRACE_PROGRESS_EMA_BETA")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .map(|v| v.clamp(0.0, 0.999))
        .unwrap_or(cfg.energy_progress_ema_beta);
    cfg.timing_weight = env::var("NPNR_OT_TRACE_TIMING_WEIGHT")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0);
    cfg.report_interval = env::var("NPNR_OT_TRACE_REPORT_INTERVAL")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1)
        .max(1);
    if let Some(num_threads) = env::var("NPNR_OT_TRACE_NUM_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
    {
        cfg.num_threads = num_threads.max(1);
    }
    if let Some(batch_size) = env::var("NPNR_OT_TRACE_NET_BATCH_SIZE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
    {
        cfg.net_parallel_batch_size = batch_size.max(1);
    }
    PlacerOptTrans.place(&mut ctx, &cfg).expect("place");
}
