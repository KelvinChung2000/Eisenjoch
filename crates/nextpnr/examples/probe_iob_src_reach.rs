//! Probe what the IOB at (0,183) — the tm3_vidin_cref source — can
//! actually reach via downhill pips, and how far. Tells us whether
//! beam_fail "tree=1" is a database connectivity gap or a search
//! algorithm limitation.

use nextpnr::chipdb::{ChipDb, WireId};
use nextpnr::read_packed;
use rustc_hash::FxHashSet;
use std::collections::VecDeque;
use std::path::Path;

fn main() {
    let p = "/home/kelvin/side-project/eisenjoch/chip_database/xc7_large.bin";
    let db = ChipDb::load(Path::new(p)).expect("load");

    // Find IOB at (0, 183).
    let src_tile = 183 * db.width();
    println!(
        "src tile @ (0,183) = {}, tt = {}",
        src_tile,
        db.tile_type_name(src_tile),
    );

    // List all output wires of IOBs in this tile.
    let tt = db.tile_type(src_tile);
    let bels = tt.bels.get();
    println!("\nBEL output wires in IOB tile:");
    let mut iob_out_wires: Vec<WireId> = Vec::new();
    for bel in bels {
        let bt: i32 = bel.bel_type();
        let Some(name) = db.constid_str(bt) else {
            continue;
        };
        for pin in bel.pins.get() {
            if pin.dir() != 1 {
                continue;
            }
            let wi = pin.wire();
            if wi < 0 {
                continue;
            }
            let w = WireId::new(src_tile, wi);
            let info = db.wire_info(w);
            let nid: i32 = unsafe { read_packed!(*info, name) };
            let wname = db.constid_str(nid).unwrap_or("?");
            let n_down = info.pips_downhill.get().len();
            let n_up = info.pips_uphill.get().len();
            println!("  bel={name:<6} pin_wire={wname} n_down={n_down} n_up={n_up}");
            iob_out_wires.push(w);
        }
    }

    if iob_out_wires.is_empty() {
        println!("No output wires found, aborting.");
        return;
    }

    // List the 52 downhill destinations from IO0_O.
    let io0 = iob_out_wires[0];
    let io0_info = db.wire_info(io0);
    println!(
        "\nIO0_O downhill destinations (first 20 of {}):",
        io0_info.pips_downhill.get().len()
    );
    for (i, &pip) in io0_info.pips_downhill.get().iter().take(20).enumerate() {
        let pid = nextpnr::chipdb::PipId::new(io0.tile(), pip);
        let dst = db.pip_dst_wire(pid);
        let dinfo = db.wire_info(dst);
        let dnid: i32 = unsafe { read_packed!(*dinfo, name) };
        let dname = db.constid_str(dnid).unwrap_or("?");
        let (dx, dy) = db.tile_xy(dst.tile());
        let n_peers_other_tile = {
            let mut c = 0;
            db.node_wires_cb(dst, |nw| {
                if nw.tile() != dst.tile() {
                    c += 1;
                }
            });
            c
        };
        println!(
            "  [{i:>2}] -> wire {dname} @ tile=({dx},{dy}) tt={} n_down={} peers_other_tile={n_peers_other_tile}",
            db.tile_type_name(dst.tile()),
            dinfo.pips_downhill.get().len(),
        );
    }

    // What's in the neighbour tile to the right of (0,183)?
    let neigh_tile = 183 * db.width() + 1;
    println!(
        "\nNeighbour tile (1,183) tt = {}",
        db.tile_type_name(neigh_tile),
    );
    let nt = db.tile_type(neigh_tile);
    println!(
        "  bels = {}, wires = {}, pips = {}",
        nt.bels.get().len(),
        nt.wires.get().len(),
        nt.pips.get().len()
    );

    // Trace 2-hop path: IO0_O -> LOGIC_OUTS_L_B0 -> ? and check tileconn peers.
    println!("\n=== 2-hop trace from INT_INTERFACE_LOGIC_OUTS_L_B0 ===");
    let iob_tt = db.tile_type(src_tile);
    let mut log_outs_b0_wi: i32 = -1;
    let mut log_outs_l0_wi: i32 = -1;
    for (wi, w) in iob_tt.wires.get().iter().enumerate() {
        let nid: i32 = unsafe { read_packed!(*w, name) };
        let n = db.constid_str(nid).unwrap_or("");
        if n == "INT_INTERFACE_LOGIC_OUTS_L_B0" {
            log_outs_b0_wi = wi as i32;
        }
        if n == "INT_INTERFACE_LOGIC_OUTS_L0" {
            log_outs_l0_wi = wi as i32;
        }
    }
    println!("  LOGIC_OUTS_L_B0 wi = {log_outs_b0_wi}");
    println!("  LOGIC_OUTS_L0   wi = {log_outs_l0_wi}");

    if log_outs_b0_wi >= 0 {
        let w = WireId::new(src_tile, log_outs_b0_wi);
        let info = db.wire_info(w);
        println!(
            "  LOGIC_OUTS_L_B0 has {} downhill pips, {} peers in other tile",
            info.pips_downhill.get().len(),
            {
                let mut c = 0;
                db.node_wires_cb(w, |nw| {
                    if nw.tile() != w.tile() {
                        c += 1;
                    }
                });
                c
            },
        );
        for &pip in info.pips_downhill.get().iter().take(8) {
            let pid = nextpnr::chipdb::PipId::new(src_tile, pip);
            let dst = db.pip_dst_wire(pid);
            let dinfo = db.wire_info(dst);
            let dnid: i32 = unsafe { read_packed!(*dinfo, name) };
            println!(
                "    -> {} (downhill {})",
                db.constid_str(dnid).unwrap_or("?"),
                dinfo.pips_downhill.get().len()
            );
        }
    }
    if log_outs_l0_wi >= 0 {
        let w = WireId::new(src_tile, log_outs_l0_wi);
        let info = db.wire_info(w);
        let peers_other: usize = {
            let mut c = 0;
            db.node_wires_cb(w, |nw| {
                if nw.tile() != w.tile() {
                    c += 1;
                }
            });
            c
        };
        println!(
            "  LOGIC_OUTS_L0   has {} downhill pips, {} peers in other tile",
            info.pips_downhill.get().len(),
            peers_other,
        );
        if peers_other > 0 {
            db.node_wires_cb(w, |nw| {
                if nw.tile() != w.tile() {
                    let (px, py) = db.tile_xy(nw.tile());
                    let pinfo = db.wire_info(nw);
                    let pnid: i32 = unsafe { read_packed!(*pinfo, name) };
                    println!(
                        "    peer: ({px},{py}) tt={} {}",
                        db.tile_type_name(nw.tile()),
                        db.constid_str(pnid).unwrap_or("?")
                    );
                }
            });
        }
    }

    // BFS forward through pips + node-share peers from each IOB output, with
    // a depth limit. Track reach (max manhattan from src tile) and total
    // wires reached. This bounds the search to keep the probe fast.
    const HOP_LIMIT: u32 = 30;
    for src in &iob_out_wires {
        let mut visited: FxHashSet<WireId> = FxHashSet::default();
        visited.insert(*src);
        let mut queue: VecDeque<(WireId, u32)> = VecDeque::new();
        queue.push_back((*src, 0));
        // Also seed peers of src.
        db.node_wires_cb(*src, |nw| {
            if visited.insert(nw) {
                queue.push_back((nw, 0));
            }
        });
        let mut max_dx: i32 = 0;
        let mut max_dy: i32 = 0;
        let mut max_manhattan: i32 = 0;
        let (sx, sy) = db.tile_xy(src.tile());
        let update_dist =
            |w: WireId, max_dx: &mut i32, max_dy: &mut i32, max_manhattan: &mut i32| {
                let (tx, ty) = db.tile_xy(w.tile());
                let dx = (tx - sx).abs();
                let dy = (ty - sy).abs();
                if dx > *max_dx {
                    *max_dx = dx;
                }
                if dy > *max_dy {
                    *max_dy = dy;
                }
                if dx + dy > *max_manhattan {
                    *max_manhattan = dx + dy;
                }
            };
        while let Some((w, hop)) = queue.pop_front() {
            if hop >= HOP_LIMIT {
                continue;
            }
            update_dist(w, &mut max_dx, &mut max_dy, &mut max_manhattan);
            let info = db.wire_info(w);
            // Expand through downhill pips.
            for &pip in info.pips_downhill.get() {
                let pid = nextpnr::chipdb::PipId::new(w.tile(), pip);
                let dst = db.pip_dst_wire(pid);
                if visited.insert(dst) {
                    queue.push_back((dst, hop + 1));
                }
            }
            // Expand through tileconn (node-share peers).
            db.node_wires_cb(w, |nw| {
                if visited.insert(nw) {
                    queue.push_back((nw, hop + 1));
                }
            });
        }
        let info = db.wire_info(*src);
        let nid: i32 = unsafe { read_packed!(*info, name) };
        let nm = db.constid_str(nid).unwrap_or("?");
        println!(
            "\n=== From {nm}: BFS up to {HOP_LIMIT} hops ===\n  visited wires = {}\n  max_dx = {max_dx}, max_dy = {max_dy}, max_manhattan = {max_manhattan}",
            visited.len(),
        );
    }
}
