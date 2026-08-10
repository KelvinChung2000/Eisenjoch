//! Reverse-BFS from a CLBLM_M_CLK sink to the BUFG source, enumerating
//! pips_uphill at each layer. Tells us whether the "last hop" into the
//! CLB actually connects to the GCLK spine we just built.

use nextpnr::chipdb::{ChipDb, PipId, WireId};
use rustc_hash::FxHashSet;
use std::collections::VecDeque;
use std::path::Path;

fn describe(chipdb: &ChipDb, wire: WireId) -> String {
    let info = chipdb.wire_info(wire);
    let name_id: i32 = unsafe { nextpnr::read_packed!(*info, name) };
    let wname = chipdb.constid_str(name_id).unwrap_or("<anon>").to_string();
    let wtype = chipdb.wire_type(wire).to_string();
    let (x, y) = chipdb.tile_xy(wire.tile());
    let nid = chipdb.node_id(wire);
    let n_peers = {
        let mut n = 0usize;
        chipdb.node_wires_cb(wire, |_| n += 1);
        n
    };
    format!(
        "{}({}:{}) [{},{}] type={} nid={:?} peers={}",
        wname,
        wire.tile(),
        wire.index(),
        x,
        y,
        wtype,
        nid,
        n_peers
    )
}

fn main() {
    let chipdb_path = std::env::args().nth(1).unwrap_or_else(|| {
        "/home/kelvin/side-project/eisenjoch/chip_database/xc7_large.bin".into()
    });
    let db = ChipDb::load(Path::new(&chipdb_path)).expect("load chipdb");

    let target_tile: i32 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(35962);
    let target_widx: i32 = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1726);
    let max_layers: usize = std::env::args()
        .nth(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or(6);

    let sink = WireId::new(target_tile, target_widx);
    println!("Sink: {}", describe(&db, sink));

    let mut seen: FxHashSet<WireId> = FxHashSet::default();
    let mut frontier: VecDeque<(WireId, usize)> = VecDeque::new();
    frontier.push_back((sink, 0));
    seen.insert(sink);

    while let Some((w, depth)) = frontier.pop_front() {
        if depth >= max_layers {
            continue;
        }
        let info = db.wire_info(w);
        let up_pips = info.pips_uphill.get().to_vec();

        // Also expand node peers (free hops).
        let mut peers = Vec::new();
        db.node_wires_cb(w, |p| {
            if p != w {
                peers.push(p);
            }
        });

        // Print per-wire summary (only first few peers to keep log small).
        let nid = db.node_id(w);
        println!("\n[d={}] {}", depth, describe(&db, w));
        if up_pips.is_empty() && peers.is_empty() && depth > 0 {
            println!("    (dead end: no uphill, no peers)");
        }
        if up_pips.len() > 20 {
            println!("    {} uphill pips (truncated)", up_pips.len());
        } else {
            for &pidx in &up_pips {
                let pip = PipId::new(w.tile(), pidx);
                let src = db.pip_src_wire(pip);
                println!("    <- {}", describe(&db, src));
                if seen.insert(src) {
                    frontier.push_back((src, depth + 1));
                }
            }
        }
        if peers.len() > 8 {
            println!(
                "    {} node peers (sample: {}, {}, {})",
                peers.len(),
                describe(&db, peers[0]),
                describe(&db, peers[1]),
                describe(&db, peers[peers.len() - 1])
            );
            // BFS only one representative peer at this depth
            if seen.insert(peers[0]) {
                frontier.push_back((peers[0], depth));
            }
        } else {
            for &p in &peers {
                println!("    ~= {}", describe(&db, p));
                if seen.insert(p) {
                    frontier.push_back((p, depth));
                }
            }
        }
        let _ = nid;
    }
}
