//! Diagnose why the 5 BUFG-related nets on sv3 cannot be routed.
//!
//! After packing + placement, locate the failing nets by name and report
//! for each one:
//!   - source wire, wire_type, #pips_downhill, node peer count
//!   - per-sink: wire name/type, #pips_uphill, node peer count
//!   - whether source and any sink share a node_id (immediate bind)
//!   - unbounded Dijkstra from src's whole node → does sink get reached?
//!     number of PIP hops on the optimal path.
//!
//! This tells us, for each of {tm3_clk_v0, tm3_clk_v2, $signal$141,
//! $signal$310, $signal$311}, whether the chipdb provides a path or not.

use nextpnr::chipdb::{ChipDb, PipId, WireId};
use nextpnr::context::Context;
use nextpnr::frontend::parse_json;
use nextpnr::packer;
use nextpnr::placer::opt_trans::{place_opt_trans, OptTransPlacerCfg};
use nextpnr::router::astar::{astar_search, default_pip_cost, AStarOptions, PathCostModel};
use nextpnr::router::common::{collect_sink_wires, resolve_source_wire};
use nextpnr::timing::DelayT;
use rustc_hash::FxHashSet;
use std::path::Path;

struct DijkstraModel;

impl PathCostModel for DijkstraModel {
    fn pip_cost(&self, ctx: &Context, pip: PipId) -> DelayT {
        default_pip_cost(ctx, pip)
    }
    fn heuristic(&self, _ctx: &Context, _wire: WireId, _dst: WireId) -> DelayT {
        0
    }
}

fn describe_wire(chipdb: &ChipDb, wire: WireId) -> String {
    let info = chipdb.wire_info(wire);
    let name_id: i32 = unsafe { nextpnr::read_packed!(*info, name) };
    let wname = chipdb.constid_str(name_id).unwrap_or("<anon>").to_string();
    let wtype = chipdb.wire_type(wire).to_string();
    let (x, y) = chipdb.tile_xy(wire.tile());
    format!(
        "{}({}:{}) [{},{}] type={}",
        wname,
        wire.tile(),
        wire.index(),
        x,
        y,
        wtype
    )
}

fn count_node_peers(chipdb: &ChipDb, wire: WireId) -> usize {
    let mut n = 0usize;
    chipdb.node_wires_cb(wire, |_| n += 1);
    n
}

fn dijkstra_reach(
    ctx: &Context,
    src: WireId,
    dst: WireId,
    visit_limit: usize,
) -> Option<Vec<PipId>> {
    let chipdb = ctx.chipdb();
    let mut src_set: FxHashSet<WireId> = FxHashSet::default();
    src_set.insert(src);
    chipdb.node_wires_cb(src, |nw| {
        src_set.insert(nw);
    });
    let opts = AStarOptions {
        visit_limit: Some(visit_limit),
        exhaustive: false,
        retain_trace: false,
        stop_on_first_touch: false,
    };
    astar_search(ctx, &DijkstraModel, &src_set, dst, &opts).path
}

fn main() {
    let chipdb_path = std::env::args().nth(1).unwrap_or_else(|| {
        "/home/kelvin/side-project/eisenjoch/chip_database/xc7_large.bin".into()
    });
    let design_path = std::env::args().nth(2).unwrap_or_else(|| {
        "/home/kelvin/side-project/eisenjoch/benchmark/output/stereovision3.json".into()
    });

    let db = ChipDb::load(Path::new(&chipdb_path)).expect("load chipdb");
    let mut ctx = Context::new(db);
    let json = std::fs::read_to_string(&design_path).expect("read design");
    ctx.design = parse_json(&json, &ctx.id_pool).expect("parse design");
    packer::pack(&mut ctx, None).expect("pack");

    let mut pcfg = OptTransPlacerCfg::default();
    pcfg.max_outer_iters = 15;
    pcfg.num_threads = 1;
    place_opt_trans(&mut ctx, &pcfg).expect("place_opt_trans");

    let targets = [
        "tm3_clk_v2",
        "tm3_clk_v0",
        "$signal$141",
        "$signal$310",
        "$signal$311",
    ];

    // Reach budget for unbounded A* probe. With 65k-peer clock nodes the
    // frontier starts pre-seeded; a couple hundred k pops should be enough
    // to decide reachability within a few PIP hops.
    let visit_limit: usize = std::env::var("PROBE_VISIT_LIMIT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(500_000);

    for name in targets {
        let net_idx = match ctx.design.iter_net_indices().find(|&n| {
            let net = ctx.net(n);
            ctx.name_of(net.name_id()) == name
        }) {
            Some(n) => n,
            None => {
                println!("=== {}: NOT FOUND in design", name);
                continue;
            }
        };

        let net = ctx.net(net_idx);
        if !net.is_alive() {
            println!("=== {}: dead", name);
            continue;
        }

        println!("\n========== {} ==========", name);
        println!(
            "  users={} wires_in_tree={}",
            net.num_users(),
            net.wires().len()
        );

        let src_wire = match resolve_source_wire(&ctx, net_idx) {
            Ok(Some(w)) => w,
            Ok(None) => {
                println!("  source: <none> (constant net?)");
                continue;
            }
            Err(e) => {
                println!("  source resolve error: {:?}", e);
                continue;
            }
        };

        let chipdb = ctx.chipdb();
        let src_info = chipdb.wire_info(src_wire);
        let src_down = src_info.pips_downhill.get().len();
        let src_up = src_info.pips_uphill.get().len();
        let src_peers = count_node_peers(chipdb, src_wire);
        println!("  SRC: {}", describe_wire(chipdb, src_wire));
        println!(
            "       pips_downhill={} pips_uphill={} node_peers={}",
            src_down, src_up, src_peers
        );

        let sinks = collect_sink_wires(&ctx, net_idx);
        println!("  {} sink wires", sinks.len());

        // Examine up to first N sinks and probe reachability.
        let probe_sinks = sinks.iter().take(3).copied().collect::<Vec<_>>();
        for (i, sink) in probe_sinks.iter().enumerate() {
            let sink_info = chipdb.wire_info(*sink);
            let sink_down = sink_info.pips_downhill.get().len();
            let sink_up = sink_info.pips_uphill.get().len();
            let sink_peers = count_node_peers(chipdb, *sink);
            println!("  SINK[{}]: {}", i, describe_wire(chipdb, *sink));
            println!(
                "           pips_downhill={} pips_uphill={} node_peers={}",
                sink_down, sink_up, sink_peers
            );

            // Shared-node check (bind-immediate).
            let src_nid = chipdb.node_id(src_wire);
            let sink_nid = chipdb.node_id(*sink);
            let shared_node = src_nid.is_some() && src_nid == sink_nid;
            println!(
                "           src_node_id={:?} sink_node_id={:?} shared={}",
                src_nid, sink_nid, shared_node
            );

            // Unbounded-ish Dijkstra probe.
            let t = std::time::Instant::now();
            match dijkstra_reach(&ctx, src_wire, *sink, visit_limit) {
                Some(pips) => println!(
                    "           REACHABLE in {} PIPs (search took {:.2} ms)",
                    pips.len(),
                    t.elapsed().as_secs_f64() * 1000.0
                ),
                None => println!(
                    "           UNREACHABLE within {} pops ({:.2} ms)",
                    visit_limit,
                    t.elapsed().as_secs_f64() * 1000.0
                ),
            }

            // Diagnostic: enumerate the 3 pips_uphill sources of this sink.
            // Tells us WHICH wires drive the sink, so we can check whether
            // they reside on the BUFG source's node.
            let sink_up_pips: Vec<PipId> = sink_info
                .pips_uphill
                .get()
                .iter()
                .map(|&idx| PipId::new(sink.tile(), idx))
                .collect();
            for (j, &pip) in sink_up_pips.iter().enumerate() {
                let src_w = chipdb.pip_src_wire(pip);
                let src_nid_of_driver = chipdb.node_id(src_w);
                println!(
                    "             uphill_src[{}]: {} src_nid={:?} same_as_net_src_node={}",
                    j,
                    describe_wire(chipdb, src_w),
                    src_nid_of_driver,
                    src_nid_of_driver.is_some() && src_nid_of_driver == chipdb.node_id(src_wire)
                );
            }
        }

        // Source-node connectivity summary: count total pips_downhill
        // summed over all peers of the source's node. If zero, the chipdb
        // has no way to leave this electrical node.
        let mut total_src_node_downhill: usize = 0;
        let mut total_src_node_uphill: usize = 0;
        let mut src_peer_sample: Vec<WireId> = Vec::new();
        chipdb.node_wires_cb(src_wire, |peer| {
            let info = chipdb.wire_info(peer);
            total_src_node_downhill += info.pips_downhill.get().len();
            total_src_node_uphill += info.pips_uphill.get().len();
            if src_peer_sample.len() < 3 {
                src_peer_sample.push(peer);
            }
        });
        let self_down = src_info.pips_downhill.get().len();
        let self_up = src_info.pips_uphill.get().len();
        println!(
            "  SRC_NODE summary: total_downhill={} total_uphill={} (includes self_down={} self_up={})",
            total_src_node_downhill + self_down,
            total_src_node_uphill + self_up,
            self_down,
            self_up,
        );
        for (i, peer) in src_peer_sample.iter().enumerate() {
            let info = chipdb.wire_info(*peer);
            println!(
                "    src_peer[{}]: {} pips_down={} pips_up={}",
                i,
                describe_wire(chipdb, *peer),
                info.pips_downhill.get().len(),
                info.pips_uphill.get().len(),
            );
        }
    }
}
