//! Rank sites on the nearest candidate tile by local connectivity and competition for routing
//! nodes one programmable switch away from each connected BEL pin.

use super::common::DriverNodeRegistry;
use crate::chipdb::{BelId, WireId};
use crate::context::Context;
use crate::netlist::{CellId, CellPin, NetId};
use rustc_hash::FxHashMap;

/// Compare short physical paths between two pins. Beyond two programmable
/// switches the global route matters more than this local lookahead.
fn local_hops(
    ctx: &Context,
    registry: &DriverNodeRegistry,
    net: NetId,
    from: WireId,
    to: WireId,
) -> u32 {
    let db = ctx.chipdb();
    if from == to {
        return 0;
    }
    let mut downhill = Vec::new();
    db.pips_downhill_cb(from, |p| {
        let wire = db.pip_dst_wire(p);
        if registry.lookup(ctx, wire).is_none_or(|owner| owner == net) {
            downhill.push(wire);
        }
    });
    if downhill.contains(&to) {
        return 1;
    }
    let mut hops = 3;
    db.pips_uphill_cb(to, |p| {
        if downhill.contains(&db.pip_src_wire(p)) {
            hops = 2;
        }
    });
    hops
}

fn rank_pin_access(
    ctx: &Context,
    registry: &DriverNodeRegistry,
    cell: CellId,
    options: &mut [(BelId, u32)],
    demand: &AccessDemand,
) {
    if options.len() < 2 {
        return;
    }
    let tile = options[0].0.tile();
    let mut neighbors = Vec::new();
    for (&port, data) in ctx.design.cell(cell).ports() {
        let Some(net_id) = data.net() else { continue };
        let net = ctx.net(net_id);
        let Some(driver) = net.driver_cell_port() else {
            continue;
        };
        let outgoing = driver == CellPin::new(cell, port);
        let mut record = |other: CellPin| {
            if !other.is_valid() {
                return;
            }
            if let Some(bel) = ctx.design.cell(other.cell).bel {
                if bel.tile() == tile {
                    if let Some(w) = ctx.bel(bel).pin_wire(other.port) {
                        neighbors.push((port, net_id, outgoing, w.id()));
                    }
                }
            }
        };
        if outgoing {
            for &user in net.users() {
                record(user);
            }
        } else {
            record(driver);
        }
    }
    options.sort_by_cached_key(|&(bel, _)| {
        let hops = neighbors
            .iter()
            .map(|&(port, net, outgoing, other)| {
                let Some(pin) = ctx.bel(bel).pin_wire(port) else {
                    return 4u64;
                };
                let (from, to) = if outgoing {
                    (pin.id(), other)
                } else {
                    (other, pin.id())
                };
                u64::from(local_hops(ctx, registry, net, from, to))
            })
            .sum::<u64>();
        (hops, demand.cost(ctx, cell, bel).to_bits())
    });
}

/// Expected contention at the first switch outside each placed pin. A pin
/// distributes one unit across its distinct access wires. A net may share
/// a wire between branches, so retain its largest claim on that wire.
#[derive(Default)]
pub(super) struct AccessDemand {
    total: FxHashMap<WireId, f64>,
    by_net: FxHashMap<(WireId, NetId), f64>,
}

impl AccessDemand {
    pub(super) fn seed_from_bound(ctx: &Context) -> Self {
        let mut demand = Self::default();
        for (cell, data) in ctx.design.iter_alive_cells() {
            if let Some(bel) = data.bel {
                demand.record(ctx, cell, bel);
            }
        }
        demand
    }

    fn visit(ctx: &Context, cell: CellId, bel: BelId, mut visit: impl FnMut(WireId, NetId, f64)) {
        let db = ctx.chipdb();
        let mut choices = Vec::new();
        for (&port, data) in ctx.design.cell(cell).ports() {
            let Some(net) = data.net() else { continue };
            let Some(driver) = ctx.net(net).driver_cell_port() else {
                continue;
            };
            if ctx.net(net).num_users() == 0 {
                continue;
            }
            let Some(pin) = ctx.bel(bel).pin_wire(port) else {
                continue;
            };
            choices.clear();
            if driver == CellPin::new(cell, port) {
                db.pips_downhill_cb(pin.id(), |p| choices.push(db.pip_dst_wire(p)));
            } else {
                db.pips_uphill_cb(pin.id(), |p| choices.push(db.pip_src_wire(p)));
            }
            choices.sort_unstable_by_key(|w| (w.tile(), w.index()));
            choices.dedup();
            let share = 1.0 / choices.len().max(1) as f64;
            for &wire in &choices {
                visit(wire, net, share);
            }
        }
    }

    pub(super) fn record(&mut self, ctx: &Context, cell: CellId, bel: BelId) {
        Self::visit(ctx, cell, bel, |wire, net, share| {
            let own = self.by_net.entry((wire, net)).or_default();
            if share > *own {
                *self.total.entry(wire).or_default() += share - *own;
                *own = share;
            }
        });
    }

    fn cost(&self, ctx: &Context, cell: CellId, bel: BelId) -> f64 {
        let mut cost = 0.0;
        Self::visit(ctx, cell, bel, |wire, net, share| {
            let own = self.by_net.get(&(wire, net)).copied().unwrap_or_default();
            let total = self.total.get(&wire).copied().unwrap_or_default();
            cost += (share - own).max(0.0) * (total - own);
        });
        cost
    }

    pub(super) fn rank_nearest_tile(
        &self,
        ctx: &Context,
        registry: &DriverNodeRegistry,
        cell: CellId,
        candidates: &mut std::iter::Peekable<impl Iterator<Item = (BelId, u32)>>,
    ) -> Vec<(BelId, u32)> {
        let mut nearby = Vec::new();
        if let Some(&(first, _)) = candidates.peek() {
            // Keep the transport solution's tile priority. First-switch
            // pressure measures local competition and cannot price the new
            // inter-tile traffic created by moving a cell to another tile.
            while let Some(slot) = candidates.next_if(|(bel, _)| bel.tile() == first.tile()) {
                nearby.push(slot);
            }
            rank_pin_access(ctx, registry, cell, &mut nearby, self);
        }

        nearby
    }
}
