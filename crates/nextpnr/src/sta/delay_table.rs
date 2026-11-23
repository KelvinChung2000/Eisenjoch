use std::cmp::Reverse;

use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

use super::interconnect::{DelayBounds, PipDelays, SearchGoal};
use crate::chipdb::{ChipDb, WireId};
use crate::netlist::PortType;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PinWire {
    pub tile_type: i32,
    pub wire: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TileOffset {
    pub dx: i32,
    pub dy: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TableKey {
    pub source: PinWire,
    pub sink: PinWire,
    pub offset: TileOffset,
}

impl TableKey {
    pub fn between(chipdb: &ChipDb, source: WireId, sink: WireId) -> Self {
        let (sx, sy) = chipdb.tile_xy(source.tile());
        let (tx, ty) = chipdb.tile_xy(sink.tile());
        Self {
            source: PinWire::of(chipdb, source),
            sink: PinWire::of(chipdb, sink),
            offset: TileOffset {
                dx: tx - sx,
                dy: ty - sy,
            },
        }
    }
}

impl PinWire {
    fn of(chipdb: &ChipDb, wire: WireId) -> Self {
        Self {
            tile_type: chipdb.tile_type_index(wire.tile()),
            wire: wire.index(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reach {
    Delay(DelayBounds),
    Unreachable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableEntry {
    Sampled { sample: i32, reach: Reach },
    VariesWithSource,
}

#[derive(Debug, Clone)]
pub struct DelayTable {
    entries: FxHashMap<TableKey, TableEntry>,
}

impl DelayTable {
    pub(super) fn build(chipdb: &ChipDb, pips: &PipDelays) -> Self {
        let grid = Grid::new(chipdb);
        let searches: Vec<(i32, i32)> = (0..grid.pins.len())
            .filter(|&tile_type| !grid.pins[tile_type].sources.is_empty())
            .flat_map(|tile_type| {
                let sources = &grid.pins[tile_type].sources;
                grid.samples(tile_type)
                    .into_iter()
                    .flat_map(move |sample| sources.iter().map(move |&wire| (sample, wire)))
            })
            .collect();
        let found: Vec<Vec<(TableKey, i64, i32, Reach)>> = searches
            .par_iter()
            .map(|&(sample, wire)| {
                let source = WireId::new(sample, wire);
                let settled = pips.shortest_delays(chipdb, source, SearchGoal::AllWires);
                grid.sink_pins()
                    .map(|sink| {
                        let reach = settled
                            .get(&sink)
                            .map_or(Reach::Unreachable, |&delay| Reach::Delay(delay));
                        (
                            TableKey::between(chipdb, source, sink),
                            grid.pair_eccentricity(sample, sink.tile()),
                            sample,
                            reach,
                        )
                    })
                    .collect()
            })
            .collect();
        let mut entries: FxHashMap<TableKey, (i64, TableEntry)> = FxHashMap::default();
        for (key, eccentricity, sample, reach) in found.into_iter().flatten() {
            let sampled = TableEntry::Sampled { sample, reach };
            entries
                .entry(key)
                .and_modify(|kept| match kept.1 {
                    TableEntry::Sampled {
                        reach: kept_reach, ..
                    } if kept_reach != reach => {
                        kept.1 = TableEntry::VariesWithSource;
                    }
                    TableEntry::Sampled { .. } if eccentricity < kept.0 => {
                        *kept = (eccentricity, sampled);
                    }
                    TableEntry::Sampled { .. } | TableEntry::VariesWithSource => {}
                })
                .or_insert((eccentricity, sampled));
        }
        let varying_pins: FxHashSet<(PinWire, PinWire)> = entries
            .iter()
            .filter(|(_, (_, entry))| *entry == TableEntry::VariesWithSource)
            .map(|(key, _)| (key.source, key.sink))
            .collect();
        Self {
            entries: entries
                .into_iter()
                .map(
                    |(key, (_, entry))| match varying_pins.contains(&(key.source, key.sink)) {
                        true => (key, TableEntry::VariesWithSource),
                        false => (key, entry),
                    },
                )
                .collect(),
        }
    }

    pub fn get(&self, key: &TableKey) -> Option<TableEntry> {
        self.entries.get(key).copied()
    }

    pub fn entries(&self) -> impl Iterator<Item = (&TableKey, &TableEntry)> {
        self.entries.iter()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[derive(Debug, Default)]
struct PinWires {
    sources: Vec<i32>,
    sinks: Vec<i32>,
}

struct Grid {
    width: i32,
    height: i32,
    tile_types: Vec<usize>,
    pins: Vec<PinWires>,
    sink_tiles: Vec<i32>,
}

impl Grid {
    fn new(chipdb: &ChipDb) -> Self {
        let tile_types: Vec<usize> = (0..chipdb.num_tiles())
            .map(|tile| {
                usize::try_from(chipdb.tile_type_index(tile)).expect("tile type index is valid")
            })
            .collect();
        let type_count = tile_types.iter().max().map_or(0, |&last| last + 1);
        let pins: Vec<PinWires> = (0..type_count)
            .map(|tile_type| {
                let tile_type = i32::try_from(tile_type).expect("tile type index fits in i32");
                let mut sources = FxHashSet::default();
                let mut sinks = FxHashSet::default();
                for bel in chipdb.tile_type_by_index(tile_type).bels.get() {
                    for pin in bel.pins.get() {
                        let drives = pin.dir() != PortType::In as i32;
                        let loads = pin.dir() != PortType::Out as i32;
                        if drives {
                            sources.insert(pin.wire());
                        }
                        if loads {
                            sinks.insert(pin.wire());
                        }
                    }
                }
                let sorted = |set: FxHashSet<i32>| {
                    let mut wires: Vec<i32> = set.into_iter().collect();
                    wires.sort_unstable();
                    wires
                };
                PinWires {
                    sources: sorted(sources),
                    sinks: sorted(sinks),
                }
            })
            .collect();
        let sink_tiles = (0..chipdb.num_tiles())
            .filter(|&tile| !pins[tile_types[tile as usize]].sinks.is_empty())
            .collect();
        Self {
            width: chipdb.width(),
            height: chipdb.height(),
            tile_types,
            pins,
            sink_tiles,
        }
    }

    fn xy(&self, tile: i32) -> (i32, i32) {
        (tile % self.width, tile / self.width)
    }

    fn sink_pins(&self) -> impl Iterator<Item = WireId> + '_ {
        self.sink_tiles.iter().flat_map(move |&tile| {
            self.pins[self.tile_types[tile as usize]]
                .sinks
                .iter()
                .map(move |&wire| WireId::new(tile, wire))
        })
    }

    fn pair_eccentricity(&self, source: i32, sink: i32) -> i64 {
        let ((sx, sy), (tx, ty)) = (self.xy(source), self.xy(sink));
        let ex = i64::from(sx + tx - (self.width - 1));
        let ey = i64::from(sy + ty - (self.height - 1));
        ex * ex + ey * ey
    }

    fn offset_class(&self, source: i32, sink: i32) -> (TileOffset, usize) {
        let ((sx, sy), (tx, ty)) = (self.xy(source), self.xy(sink));
        (
            TileOffset {
                dx: tx - sx,
                dy: ty - sy,
            },
            self.tile_types[sink as usize],
        )
    }

    fn samples(&self, tile_type: usize) -> Vec<i32> {
        let candidates: Vec<i32> = (0..self.tile_types.len() as i32)
            .filter(|&tile| self.tile_types[tile as usize] == tile_type)
            .collect();
        let covers = |tile: i32| {
            self.sink_tiles
                .iter()
                .map(move |&sink| self.offset_class(tile, sink))
        };
        let mut uncovered: FxHashSet<(TileOffset, usize)> =
            candidates.iter().flat_map(|&tile| covers(tile)).collect();
        let mut chosen = Vec::new();
        while !uncovered.is_empty() {
            let sample = candidates
                .iter()
                .copied()
                .filter(|tile| !chosen.contains(tile))
                .max_by_key(|&tile| {
                    (
                        covers(tile)
                            .filter(|class| uncovered.contains(class))
                            .count(),
                        Reverse(self.pair_eccentricity(tile, tile)),
                        Reverse(tile),
                    )
                })
                .expect("an uncovered offset has a candidate sample that covers it");
            for class in covers(sample) {
                uncovered.remove(&class);
            }
            chosen.push(sample);
        }
        chosen
    }
}
