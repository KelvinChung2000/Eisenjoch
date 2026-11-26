use std::borrow::Borrow;
use std::fmt;
use std::hash::Hash;

use rustc_hash::FxHashMap;

use super::{BelId, ChipDb, PipId, TileTypePod, WireId};
use crate::read_packed;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NameKind {
    Tile,
    Bel,
    Wire,
    Pip,
}

impl fmt::Display for NameKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Tile => "tile",
            Self::Bel => "BEL",
            Self::Wire => "wire",
            Self::Pip => "pip",
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum NameError {
    #[error("{kind} name {name:?} has {found} '/'-separated segments, not {expected}")]
    SegmentCount {
        kind: NameKind,
        name: String,
        found: usize,
        expected: usize,
    },
    #[error("pip name {name:?} names an inverting pip, which the chipdb reader does not model")]
    InvertingPip { name: String },
    #[error("{kind} name {name:?}: the chipdb has no {segment_kind} {segment:?}")]
    Unknown {
        kind: NameKind,
        name: String,
        segment_kind: NameKind,
        segment: String,
    },
    #[error("{kind} name {name:?}: the chipdb has more than one {segment_kind} {segment:?}")]
    Ambiguous {
        kind: NameKind,
        name: String,
        segment_kind: NameKind,
        segment: String,
    },
}

#[derive(Clone, Copy)]
enum Slot {
    Unique(i32),
    Ambiguous,
}

struct NameMap<K>(FxHashMap<K, Slot>);

impl<K: Hash + Eq> NameMap<K> {
    fn from_entries(entries: impl IntoIterator<Item = (K, i32)>) -> Self {
        let mut map = FxHashMap::default();
        for (key, index) in entries {
            map.entry(key)
                .and_modify(|slot| *slot = Slot::Ambiguous)
                .or_insert(Slot::Unique(index));
        }
        Self(map)
    }
}

struct TileTypeNames {
    bels: NameMap<i32>,
    wires: NameMap<i32>,
    pips: NameMap<(i32, i32)>,
}

impl TileTypeNames {
    fn of(chipdb: &ChipDb, tile_type: &TileTypePod) -> Self {
        let canon = |name: i32| chipdb.constid_canon(name);
        Self {
            bels: NameMap::from_entries(
                tile_type
                    .bels
                    .get()
                    .iter()
                    .zip(0..)
                    .map(|(bel, index)| (canon(bel.name()), index)),
            ),
            wires: NameMap::from_entries(
                tile_type
                    .wires
                    .get()
                    .iter()
                    .zip(0..)
                    .map(|(wire, index)| (canon(unsafe { read_packed!(*wire, name) }), index)),
            ),
            pips: NameMap::from_entries(tile_type.pips.get().iter().zip(0..).map(
                |(pip, index)| {
                    let dst: i32 = unsafe { read_packed!(*pip, dst_wire) };
                    let src: i32 = unsafe { read_packed!(*pip, src_wire) };
                    ((dst, src), index)
                },
            )),
        }
    }
}

pub(super) struct NameIndex {
    tiles: NameMap<String>,
    tile_types: Vec<TileTypeNames>,
}

impl NameIndex {
    pub(super) fn of(chipdb: &ChipDb) -> Self {
        Self {
            tiles: NameMap::from_entries(
                (0..chipdb.num_tiles()).map(|tile| (chipdb.tile_name(tile), tile)),
            ),
            tile_types: (0..chipdb.num_tile_types())
                .map(|tile_type| {
                    TileTypeNames::of(chipdb, chipdb.tile_type_by_index(tile_type as i32))
                })
                .collect(),
        }
    }
}

struct Query<'a> {
    kind: NameKind,
    name: &'a str,
}

impl<'a> Query<'a> {
    fn segments<const N: usize>(&self) -> Result<[&'a str; N], NameError> {
        let segments: Vec<&'a str> = self.name.split('/').collect();
        let found = segments.len();
        segments.try_into().map_err(|_| NameError::SegmentCount {
            kind: self.kind,
            name: self.name.to_owned(),
            found,
            expected: N,
        })
    }

    fn find<K: Hash + Eq + Borrow<Q>, Q: Hash + Eq + ?Sized>(
        &self,
        map: &NameMap<K>,
        key: Option<&Q>,
        segment_kind: NameKind,
        segment: &str,
    ) -> Result<i32, NameError> {
        match key.and_then(|key| map.0.get(key)) {
            Some(Slot::Unique(index)) => Ok(*index),
            Some(Slot::Ambiguous) => Err(NameError::Ambiguous {
                kind: self.kind,
                name: self.name.to_owned(),
                segment_kind,
                segment: segment.to_owned(),
            }),
            None => Err(NameError::Unknown {
                kind: self.kind,
                name: self.name.to_owned(),
                segment_kind,
                segment: segment.to_owned(),
            }),
        }
    }
}

impl ChipDb {
    fn names(&self) -> &NameIndex {
        self.names.get_or_init(|| NameIndex::of(self))
    }

    fn named_tile(&self, query: &Query, segment: &str) -> Result<(i32, &TileTypeNames), NameError> {
        let tile = query.find(&self.names().tiles, Some(segment), NameKind::Tile, segment)?;
        let tile_type = &self.names().tile_types[self.tile_type_index(tile) as usize];
        Ok((tile, tile_type))
    }

    fn named_wire_index(
        &self,
        query: &Query,
        names: &TileTypeNames,
        segment: &str,
    ) -> Result<i32, NameError> {
        let constid = self.constid_for_name(segment);
        query.find(&names.wires, constid.as_ref(), NameKind::Wire, segment)
    }

    /// The BEL that upstream nextpnr names `<tile>/<bel>`.
    pub fn bel_by_name(&self, name: &str) -> Result<BelId, NameError> {
        let query = Query {
            kind: NameKind::Bel,
            name,
        };
        let [tile, bel] = query.segments()?;
        let (tile, names) = self.named_tile(&query, tile)?;
        let constid = self.constid_for_name(bel);
        let index = query.find(&names.bels, constid.as_ref(), NameKind::Bel, bel)?;
        Ok(BelId::new(tile, index))
    }

    /// The node that upstream nextpnr names `<tile>/<wire>`, as its canonical wire.
    pub fn wire_by_name(&self, name: &str) -> Result<WireId, NameError> {
        let query = Query {
            kind: NameKind::Wire,
            name,
        };
        let [tile, wire] = query.segments()?;
        let (tile, names) = self.named_tile(&query, tile)?;
        let index = self.named_wire_index(&query, names, wire)?;
        Ok(self.canonical_wire(WireId::new(tile, index)))
    }

    /// The pip that upstream nextpnr names `<tile>/<dst wire>/<src wire>`.
    pub fn pip_by_name(&self, name: &str) -> Result<PipId, NameError> {
        let query = Query {
            kind: NameKind::Pip,
            name,
        };
        if let Ok([_, _, _, "INV"]) = query.segments::<4>() {
            return Err(NameError::InvertingPip {
                name: name.to_owned(),
            });
        }
        let [tile, dst, src] = query.segments()?;
        let (tile_index, names) = self.named_tile(&query, tile)?;
        let dst_index = self.named_wire_index(&query, names, dst)?;
        let src_index = self.named_wire_index(&query, names, src)?;
        let wires = &name[tile.len() + 1..];
        let index = query.find(
            &names.pips,
            Some(&(dst_index, src_index)),
            NameKind::Pip,
            wires,
        )?;
        Ok(PipId::new(tile_index, index))
    }
}
