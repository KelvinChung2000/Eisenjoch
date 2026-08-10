//! Compare HPWL reported by HeAP and OT, both with and without GND/VCC in the
//! sum. Diagnostic for "is GND inflating our absolute numbers and the HeAP vs
//! OT comparison".

use nextpnr::chipdb::ChipDb;
use nextpnr::context::Context;
use nextpnr::frontend::parse_json;
use nextpnr::metrics::wirelength::net_hpwl;
use nextpnr::packer;
use nextpnr::placer::heap::{PlacerHeap, PlacerHeapCfg};
use nextpnr::placer::opt_trans::{OptTransPlacerCfg, PlacerOptTrans};
use nextpnr::placer::Placer;
use std::env;
use std::path::Path;
use std::time::Instant;

fn load_fresh(chipdb_path: &str, design_path: &str) -> Context {
    let db = ChipDb::load(Path::new(chipdb_path)).expect("load chipdb");
    let mut ctx = Context::new(db);
    let json = std::fs::read_to_string(design_path).expect("read design");
    ctx.design = parse_json(&json, &ctx.id_pool).expect("parse design");
    packer::pack(&mut ctx, None).expect("pack");
    ctx
}

fn position_histogram(ctx: &Context, label: &str) {
    use std::collections::BTreeMap;
    let mut xs = Vec::<i32>::new();
    let mut ys = Vec::<i32>::new();
    for (_, cell) in ctx.design.iter_alive_cells() {
        if cell.bel_strength.is_locked() {
            continue;
        }
        if let Some(bel) = cell.bel {
            let loc = ctx.bel(bel).loc();
            xs.push(loc.x);
            ys.push(loc.y);
        }
    }
    if xs.is_empty() {
        return;
    }
    xs.sort();
    ys.sort();
    let n = xs.len();
    let mnx = xs[0];
    let mxx = xs[n - 1];
    let mny = ys[0];
    let mxy = ys[n - 1];
    println!(
        "{} — {} cells, BB = {} x {} = {}",
        label,
        n,
        mxx - mnx,
        mxy - mny,
        (mxx - mnx) * (mxy - mny)
    );

    let mut xbin: BTreeMap<i32, usize> = BTreeMap::new();
    let mut ybin: BTreeMap<i32, usize> = BTreeMap::new();
    for (x, y) in xs.iter().zip(&ys) {
        *xbin.entry(x / 20).or_insert(0) += 1;
        *ybin.entry(y / 20).or_insert(0) += 1;
    }
    println!("  X histogram (20-tile bins, non-empty only):");
    for (k, c) in &xbin {
        let bar = "#".repeat((*c).min(60));
        println!(
            "    x=[{:>3},{:>3}]  {:>3}  {}",
            k * 20,
            k * 20 + 19,
            c,
            bar
        );
    }
    println!("  Y histogram (20-tile bins, non-empty only):");
    for (k, c) in &ybin {
        let bar = "#".repeat((*c).min(60));
        println!(
            "    y=[{:>3},{:>3}]  {:>3}  {}",
            k * 20,
            k * 20 + 19,
            c,
            bar
        );
    }
    println!();
}

fn hpwl_breakdown(ctx: &Context, label: &str) {
    let mut total = 0.0f64;
    let mut gnd = 0.0f64;
    let mut vcc = 0.0f64;
    let mut clocks = 0.0f64;
    let mut regular = 0.0f64;
    let mut n_nets = 0usize;
    for (net_id, net) in ctx.design.iter_alive_nets() {
        let h = net_hpwl(ctx, net_id);
        let name = ctx.name_of(net.name);
        let lower = name.to_ascii_lowercase();
        total += h;
        n_nets += 1;
        if name == "$PACKER_GND_NET" {
            gnd += h;
        } else if name == "$PACKER_VCC_NET" {
            vcc += h;
        } else if lower.contains("clk") || lower.contains("clock") {
            clocks += h;
        } else {
            regular += h;
        }
    }
    let without_const = total - gnd - vcc;
    let without_const_clocks = regular;
    println!("{}", label);
    println!("  total alive nets:       {}", n_nets);
    println!("  TOTAL HPWL:             {:>10.0}", total);
    println!(
        "  GND contribution:       {:>10.0}  ({:.1}%)",
        gnd,
        100.0 * gnd / total.max(1.0)
    );
    println!(
        "  VCC contribution:       {:>10.0}  ({:.1}%)",
        vcc,
        100.0 * vcc / total.max(1.0)
    );
    println!(
        "  clock contribution:     {:>10.0}  ({:.1}%)",
        clocks,
        100.0 * clocks / total.max(1.0)
    );
    println!(
        "  regular nets:           {:>10.0}  ({:.1}%)",
        regular,
        100.0 * regular / total.max(1.0)
    );
    println!("  HPWL excluding GND/VCC: {:>10.0}", without_const);
    println!("  HPWL regular only:      {:>10.0}", without_const_clocks);
    println!();
}

fn main() {
    let chipdb_exact = "/home/kelvin/side-project/eisenjoch/chip_database/xc7_exact.bin";
    let design = "/home/kelvin/side-project/eisenjoch/benchmark/output/stereovision3.json";

    println!("========================================");
    println!("=== chipdb: xc7_exact ===");
    println!("========================================");

    // --- HeAP ---
    let t0 = Instant::now();
    let mut ctx = load_fresh(chipdb_exact, design);
    let mut heap_cfg = PlacerHeapCfg::default();
    heap_cfg.seed = 42;
    PlacerHeap.place(&mut ctx, &heap_cfg).expect("heap place");
    let heap_ms = t0.elapsed().as_millis();
    println!("--- HeAP ({}ms) ---", heap_ms);
    hpwl_breakdown(&ctx, "HeAP final placement");
    position_histogram(&ctx, "HeAP position distribution");

    // --- OT ---
    let t0 = Instant::now();
    let mut ctx = load_fresh(chipdb_exact, design);
    let mut ot_cfg = OptTransPlacerCfg::default();
    ot_cfg.seed = 42;
    ot_cfg.num_threads = 8;
    ot_cfg.max_outer_iters = env::var("NPNR_OT_MAX_OUTER")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);
    PlacerOptTrans.place(&mut ctx, &ot_cfg).expect("ot place");
    let ot_ms = t0.elapsed().as_millis();
    println!("--- OT default ({}ms) ---", ot_ms);
    hpwl_breakdown(&ctx, "OT final placement");
    position_histogram(&ctx, "OT position distribution");
}
