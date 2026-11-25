//! Bounds, alignment and index checks every chip database passes before `ChipDb` trusts it.

use std::fmt;
use std::mem::{align_of, offset_of, size_of};

use zerocopy::error::ConvertError;
use zerocopy::{FromBytes, Immutable, KnownLayout};

use super::access::FasmNaming;
use super::pod::*;
use super::relptr::{RelPtr, RelSlice};
use super::{ChipDb, ChipDbError, CHIPDB_MAGIC, CHIPDB_VERSION};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathSegment {
    Field(&'static str),
    Index(u32),
}

/// The field a fault sits in, displayed as `chip_info.tile_types[3].pips[17].src_wire`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldPath(Vec<PathSegment>);

impl FieldPath {
    pub fn segments(&self) -> &[PathSegment] {
        &self.0
    }
}

impl fmt::Display for FieldPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (position, segment) in self.0.iter().enumerate() {
            match segment {
                PathSegment::Field(name) if position == 0 => write!(f, "{name}")?,
                PathSegment::Field(name) => write!(f, ".{name}")?,
                PathSegment::Index(index) => write!(f, "[{index}]")?,
            }
        }
        Ok(())
    }
}

/// What is wrong with the field a [`ChipDbError::Malformed`] names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fault {
    /// The bytes `start..end` the field refers to are not inside the `len`-byte buffer.
    OutOfBounds { start: i64, end: i64, len: usize },
    /// A slice of `i32` starts at a byte offset that is not a multiple of `align`.
    Misaligned { align: usize },
    /// A product of file values does not fit the type the accessors compute it in.
    Overflow,
    /// No NUL byte follows the start of the string inside the buffer.
    Unterminated,
    /// A pointer the accessors require is null.
    Null,
    /// The value is used as an index into a table of `len` entries.
    IndexOutOfRange { index: i64, len: usize },
    /// The constid names a null entry of the id table.
    MissingConstId { index: i32 },
    /// `tile_insts` must hold `width * height` tiles, both positive.
    GridMismatch {
        width: i32,
        height: i32,
        tiles: usize,
    },
    /// A node member's root wire is not marked as the root of a node.
    NotNodeRoot,
}

pub(super) struct Validated<'a> {
    pub(super) chip_info: usize,
    pub(super) uarch: &'a [u8],
}

/// Check `bytes` as a himbaechel chip database and locate its `ChipInfoPod`.
pub(super) fn validate(bytes: &[u8]) -> Result<Validated<'_>, ChipDbError> {
    let (root_ptr, _) =
        RelPtr::<ChipInfoPod>::read_from_prefix(bytes).map_err(|_| ChipDbError::TooSmall {
            size: bytes.len(),
            min: size_of::<RelPtr<ChipInfoPod>>(),
        })?;
    let v = Validator {
        bytes,
        last_nul: bytes.iter().rposition(|&byte| byte == 0),
    };
    let (chip_info, chip) = v.pointee::<ChipInfoPod>(&Site::root("root", 0), root_ptr.offset)?;
    if chip.magic != CHIPDB_MAGIC {
        return Err(ChipDbError::MagicMismatch {
            expected: CHIPDB_MAGIC as u32,
            got: chip.magic as u32,
        });
    }
    if chip.version != CHIPDB_VERSION {
        return Err(ChipDbError::VersionMismatch {
            expected: CHIPDB_VERSION,
            got: chip.version,
        });
    }
    let at = Site::root("chip_info", chip_info);
    chip.walk(&v, &at)?;
    let uarch = v.indices(&chip, &at)?;
    Ok(Validated { chip_info, uarch })
}

#[derive(Clone, Copy)]
enum Trail<'p> {
    Root(&'static str),
    Field(&'p Site<'p>, &'static str),
    Index(&'p Site<'p>, u32),
}

#[derive(Clone, Copy)]
struct Site<'p> {
    trail: Trail<'p>,
    pos: usize,
}

impl<'p> Site<'p> {
    fn root(name: &'static str, pos: usize) -> Self {
        Site {
            trail: Trail::Root(name),
            pos,
        }
    }

    fn field(&self, name: &'static str, offset: usize) -> Site<'_> {
        Site {
            trail: Trail::Field(self, name),
            pos: self.pos + offset,
        }
    }

    fn index(&self, index: u32, pos: usize) -> Site<'_> {
        Site {
            trail: Trail::Index(self, index),
            pos,
        }
    }

    fn at(&self, pos: usize) -> Site<'p> {
        Site {
            trail: self.trail,
            pos,
        }
    }

    fn fault(&self, fault: Fault) -> ChipDbError {
        ChipDbError::Malformed {
            field: self.path(),
            offset: self.pos,
            fault,
        }
    }

    fn path(&self) -> FieldPath {
        let mut segments = Vec::new();
        let mut site = self;
        loop {
            match site.trail {
                Trail::Root(name) => {
                    segments.push(PathSegment::Field(name));
                    break;
                }
                Trail::Field(parent, name) => {
                    segments.push(PathSegment::Field(name));
                    site = parent;
                }
                Trail::Index(parent, index) => {
                    segments.push(PathSegment::Index(index));
                    site = parent;
                }
            }
        }
        segments.reverse();
        FieldPath(segments)
    }
}

macro_rules! field {
    ($site:expr, $pod:ty, $field:ident) => {
        $site.field(stringify!($field), offset_of!($pod, $field))
    };
}

struct Items<'a, T> {
    start: usize,
    items: &'a [T],
}

impl<'a, T> Items<'a, T> {
    fn len(&self) -> usize {
        self.items.len()
    }

    fn sites<'s>(&self, at: &'s Site<'s>) -> impl Iterator<Item = (Site<'s>, &'a T)> + 's
    where
        'a: 's,
    {
        let positions = (self.start..).step_by(size_of::<T>());
        (0u32..)
            .zip(positions)
            .zip(self.items)
            .map(move |((index, pos), item)| (at.index(index, pos), item))
    }
}

fn index(at: &Site<'_>, value: i64, len: usize) -> Result<usize, ChipDbError> {
    usize::try_from(value)
        .ok()
        .filter(|&index| index < len)
        .ok_or_else(|| at.fault(Fault::IndexOutOfRange { index: value, len }))
}

struct Validator<'a> {
    bytes: &'a [u8],
    last_nul: Option<usize>,
}

impl<'a> Validator<'a> {
    fn target(&self, at: &Site<'_>, offset: i32) -> Result<i64, ChipDbError> {
        (at.pos as i64)
            .checked_add(i64::from(offset))
            .ok_or_else(|| at.fault(Fault::Overflow))
    }

    fn tail(&self, start: i64) -> Option<(usize, &'a [u8])> {
        let start = usize::try_from(start).ok()?;
        Some((start, self.bytes.get(start..)?))
    }

    fn out_of_bounds(&self, at: &Site<'_>, start: i64, size: u64) -> ChipDbError {
        at.fault(Fault::OutOfBounds {
            start,
            end: start.saturating_add_unsigned(size),
            len: self.bytes.len(),
        })
    }

    fn pointee<T: FromBytes>(&self, at: &Site<'_>, offset: i32) -> Result<(usize, T), ChipDbError> {
        let start = self.target(at, offset)?;
        self.tail(start)
            .and_then(|(pos, tail)| Some((pos, T::read_from_prefix(tail).ok()?.0)))
            .ok_or_else(|| self.out_of_bounds(at, start, size_of::<T>() as u64))
    }

    fn slice<T: FromBytes + KnownLayout + Immutable>(
        &self,
        at: &Site<'_>,
        slice: RelSlice<T>,
    ) -> Result<Items<'a, T>, ChipDbError> {
        let (offset, length) = (slice.offset, slice.length);
        if length == 0 {
            return Ok(Items {
                start: 0,
                items: &[],
            });
        }
        let start = self.target(at, offset)?;
        let out_of_bounds =
            || self.out_of_bounds(at, start, u64::from(length) * size_of::<T>() as u64);
        let (pos, tail) = self.tail(start).ok_or_else(out_of_bounds)?;
        let count = usize::try_from(length).map_err(|_| out_of_bounds())?;
        match <[T]>::ref_from_prefix_with_elems(tail, count) {
            Ok((items, _)) => Ok(Items { start: pos, items }),
            Err(ConvertError::Alignment(_)) => Err(at.fault(Fault::Misaligned {
                align: align_of::<T>(),
            })),
            Err(ConvertError::Size(_)) => Err(out_of_bounds()),
            Err(ConvertError::Validity(never)) => match never {},
        }
    }

    fn nested<T: Walk>(&self, at: &Site<'_>, slice: RelSlice<T>) -> Result<(), ChipDbError> {
        for (site, item) in self.slice(at, slice)?.sites(at) {
            item.walk(self, &site)?;
        }
        Ok(())
    }

    fn flat<T: FromBytes + KnownLayout + Immutable>(
        &self,
        at: &Site<'_>,
        slice: RelSlice<T>,
    ) -> Result<(), ChipDbError> {
        self.slice(at, slice).map(drop)
    }

    fn opt_nested<T: Walk>(&self, at: &Site<'_>, ptr: RelPtr<T>) -> Result<(), ChipDbError> {
        self.opt_pointee::<T>(at, ptr.offset)
    }

    fn chip_extra(&self, at: &Site<'_>, ptr: RelPtr<u8>) -> Result<(), ChipDbError> {
        self.opt_pointee::<ChipExtraDataPod>(at, ptr.offset)
    }

    fn opt_pointee<T: Walk>(&self, at: &Site<'_>, offset: i32) -> Result<(), ChipDbError> {
        if offset == 0 {
            return Ok(());
        }
        let (start, value) = self.pointee::<T>(at, offset)?;
        value.walk(self, &at.at(start))
    }

    fn blob(&self, at: &Site<'_>, ptr: RelPtr<u8>) -> Result<(), ChipDbError> {
        match ptr.offset {
            0 => Ok(()),
            offset => self.byte(at, offset).map(drop),
        }
    }

    fn byte(&self, at: &Site<'_>, offset: i32) -> Result<(usize, &'a [u8]), ChipDbError> {
        let start = self.target(at, offset)?;
        self.tail(start)
            .filter(|(_, tail)| !tail.is_empty())
            .ok_or_else(|| self.out_of_bounds(at, start, 1))
    }

    fn string(&self, at: &Site<'_>, ptr: RelPtr<u8>) -> Result<(), ChipDbError> {
        match ptr.offset {
            0 => Err(at.fault(Fault::Null)),
            _ => self.opt_string(at, ptr),
        }
    }

    fn opt_string(&self, at: &Site<'_>, ptr: RelPtr<u8>) -> Result<(), ChipDbError> {
        if ptr.offset == 0 {
            return Ok(());
        }
        let (start, _) = self.byte(at, ptr.offset)?;
        match self.last_nul {
            Some(nul) if start <= nul => Ok(()),
            _ => Err(at.fault(Fault::Unterminated)),
        }
    }

    fn strings(&self, at: &Site<'_>, slice: RelSlice<RelPtr<u8>>) -> Result<(), ChipDbError> {
        for (site, &ptr) in self.slice(at, slice)?.sites(at) {
            self.opt_string(&site, ptr)?;
        }
        Ok(())
    }

    fn c_str(&self, at: &Site<'_>, ptr: RelPtr<u8>) -> Result<&'a [u8], ChipDbError> {
        let (_, tail) = self.byte(at, ptr.offset)?;
        let len = tail.iter().position(|&byte| byte == 0);
        len.map(|len| &tail[..len])
            .ok_or_else(|| at.fault(Fault::Unterminated))
    }

    fn indices(&self, chip: &ChipInfoPod, at: &Site<'_>) -> Result<&'a [u8], ChipDbError> {
        let uarch = self.c_str(&field!(at, ChipInfoPod, uarch), chip.uarch)?;
        let constids = self.constids(chip, at)?;
        let naming = FasmNaming::of(uarch);

        let types_at = field!(at, ChipInfoPod, tile_types);
        let types = self.slice(&types_at, chip.tile_types)?;
        let mut type_wires = Vec::with_capacity(types.len());
        for (site, tile_type) in types.sites(&types_at) {
            type_wires.push(self.tile_type(&site, tile_type, naming, &constids)?);
        }

        let shapes_at = field!(at, ChipInfoPod, tile_shapes);
        let shapes = self.slice(&shapes_at, chip.tile_shapes)?;
        let mut shape_nodes = Vec::with_capacity(shapes.len());
        for (site, shape) in shapes.sites(&shapes_at) {
            let nodes_at = field!(site, TileRoutingShapePod, wire_to_node);
            shape_nodes.push(self.slice(&nodes_at, shape.wire_to_node)?);
        }

        let node_shapes_at = field!(at, ChipInfoPod, node_shapes);
        let node_shapes = self.slice(&node_shapes_at, chip.node_shapes)?;
        let mut node_members = Vec::with_capacity(node_shapes.len());
        for (site, node_shape) in node_shapes.sites(&node_shapes_at) {
            let members_at = field!(site, NodeShapePod, tile_wires);
            node_members.push(self.slice(&members_at, node_shape.tile_wires)?);
        }

        let insts_at = field!(at, ChipInfoPod, tile_insts);
        let insts = self.slice(&insts_at, chip.tile_insts)?;
        let grid = Grid::of(chip, at, insts.len())?;
        let mut tiles = Vec::with_capacity(insts.len());
        for (site, inst) in insts.sites(&insts_at) {
            let tile_type = index(
                &field!(site, TileInstPod, tile_type),
                inst.tile_type.into(),
                types.len(),
            )?;
            let shape = index(
                &field!(site, TileInstPod, shape),
                inst.shape.into(),
                shapes.len(),
            )?;
            constids.check(&field!(site, TileInstPod, name_prefix), inst.name_prefix)?;
            tiles.push(Tile {
                wires: type_wires[tile_type],
                shape,
            });
        }

        let fabric = Fabric {
            grid,
            tiles,
            shapes_at,
            shapes,
            shape_nodes,
            node_shapes_at,
            node_shapes,
            node_members,
        };
        fabric.check_nodes()?;
        Ok(uarch)
    }

    fn constids(&self, chip: &ChipInfoPod, at: &Site<'_>) -> Result<ConstIds<'a>, ChipDbError> {
        if chip.extra_constids.offset == 0 {
            return Ok(ConstIds {
                known: 0,
                embedded: &[],
            });
        }
        let ptr_at = field!(at, ChipInfoPod, extra_constids);
        let (start, data) = self.pointee::<ConstIdDataPod>(&ptr_at, chip.extra_constids.offset)?;
        let embedded = self
            .slice(
                &field!(ptr_at.at(start), ConstIdDataPod, bba_ids),
                data.bba_ids,
            )?
            .items;
        Ok(ConstIds {
            known: data.known_id_count.max(0) as usize,
            embedded,
        })
    }

    fn tile_type(
        &self,
        at: &Site<'_>,
        tile_type: &TileTypePod,
        naming: FasmNaming,
        constids: &ConstIds<'_>,
    ) -> Result<usize, ChipDbError> {
        let bels_at = field!(at, TileTypePod, bels);
        let bels = self.slice(&bels_at, tile_type.bels)?;
        let wires_at = field!(at, TileTypePod, wires);
        let wires = self.slice(&wires_at, tile_type.wires)?;
        let pips_at = field!(at, TileTypePod, pips);
        let pips = self.slice(&pips_at, tile_type.pips)?;

        for (site, bel) in bels.sites(&bels_at) {
            let pins_at = field!(site, BelDataPod, pins);
            for (pin_at, pin) in self.slice(&pins_at, bel.pins)?.sites(&pins_at) {
                index(
                    &field!(pin_at, BelPinPod, wire),
                    pin.wire.into(),
                    wires.len(),
                )?;
            }
            let extra_at = field!(site, BelDataPod, extra_data);
            self.fasm_feature(&extra_at, bel.extra_data, naming, constids)?;
        }

        for (site, wire) in wires.sites(&wires_at) {
            let uphill_at = field!(site, TileWireDataPod, pips_uphill);
            let downhill_at = field!(site, TileWireDataPod, pips_downhill);
            for (list_at, list) in [
                (uphill_at, wire.pips_uphill),
                (downhill_at, wire.pips_downhill),
            ] {
                for (pip_at, &pip) in self.slice(&list_at, list)?.sites(&list_at) {
                    index(&pip_at, pip.into(), pips.len())?;
                }
            }
            let refs_at = field!(site, TileWireDataPod, bel_pins);
            for (ref_at, bel_pin) in self.slice(&refs_at, wire.bel_pins)?.sites(&refs_at) {
                index(
                    &field!(ref_at, BelPinRefPod, bel),
                    bel_pin.bel.into(),
                    bels.len(),
                )?;
            }
        }

        for (site, pip) in pips.sites(&pips_at) {
            index(
                &field!(site, PipDataPod, src_wire),
                pip.src_wire.into(),
                wires.len(),
            )?;
            index(
                &field!(site, PipDataPod, dst_wire),
                pip.dst_wire.into(),
                wires.len(),
            )?;
            let extra_at = field!(site, PipDataPod, extra_data);
            self.fasm_feature(&extra_at, pip.extra_data, naming, constids)?;
        }
        Ok(wires.len())
    }

    fn fasm_feature(
        &self,
        at: &Site<'_>,
        extra_data: RelPtr<u8>,
        naming: FasmNaming,
        constids: &ConstIds<'_>,
    ) -> Result<(), ChipDbError> {
        match (naming, extra_data.offset) {
            (FasmNaming::FabulousNames, _) | (FasmNaming::ExtraData, 0) => Ok(()),
            (FasmNaming::ExtraData, offset) => {
                let (start, pod) = self.pointee::<FasmFeaturePod>(at, offset)?;
                constids.check(&field!(at.at(start), FasmFeaturePod, feature), pod.feature)
            }
        }
    }
}

struct ConstIds<'a> {
    known: usize,
    embedded: &'a [RelPtr<u8>],
}

impl ConstIds<'_> {
    fn check(&self, at: &Site<'_>, id: i32) -> Result<(), ChipDbError> {
        let entry = index(at, id.into(), self.known + self.embedded.len())?;
        let null = entry
            .checked_sub(self.known)
            .and_then(|embedded| self.embedded.get(embedded))
            .is_some_and(|ptr| ptr.offset == 0);
        if null {
            return Err(at.fault(Fault::MissingConstId { index: id }));
        }
        Ok(())
    }
}

struct Grid {
    width: usize,
    height: usize,
}

impl Grid {
    fn of(chip: &ChipInfoPod, at: &Site<'_>, tiles: usize) -> Result<Self, ChipDbError> {
        let (width, height) = (chip.width, chip.height);
        let area = width
            .checked_mul(height)
            .ok_or_else(|| field!(at, ChipInfoPod, width).fault(Fault::Overflow))?;
        match (
            usize::try_from(width),
            usize::try_from(height),
            usize::try_from(area),
        ) {
            (Ok(w), Ok(h), Ok(area)) if w > 0 && h > 0 && area == tiles => Ok(Grid {
                width: w,
                height: h,
            }),
            _ => Err(
                field!(at, ChipInfoPod, tile_insts).fault(Fault::GridMismatch {
                    width,
                    height,
                    tiles,
                }),
            ),
        }
    }

    fn relative(
        &self,
        (x, y): (usize, usize),
        (dx_at, dy_at): (&Site<'_>, &Site<'_>),
        dx: i16,
        dy: i16,
    ) -> Result<usize, ChipDbError> {
        match dx {
            RelNodeRefPod::MODE_ROW_CONST => Ok(y * self.width),
            RelNodeRefPod::MODE_GLB_CONST => Ok(0),
            _ => {
                let tx = index(dx_at, x as i64 + i64::from(dx), self.width)?;
                let ty = index(dy_at, y as i64 + i64::from(dy), self.height)?;
                Ok(ty * self.width + tx)
            }
        }
    }
}

struct Tile {
    wires: usize,
    shape: usize,
}

struct Fabric<'a, 'p> {
    grid: Grid,
    tiles: Vec<Tile>,
    shapes_at: Site<'p>,
    shapes: Items<'a, TileRoutingShapePod>,
    shape_nodes: Vec<Items<'a, RelNodeRefPod>>,
    node_shapes_at: Site<'p>,
    node_shapes: Items<'a, NodeShapePod>,
    node_members: Vec<Items<'a, RelTileWireRefPod>>,
}

impl Fabric<'_, '_> {
    fn check_nodes(&self) -> Result<(), ChipDbError> {
        for (tile, entry) in self.tiles.iter().enumerate() {
            let xy = (tile % self.grid.width, tile / self.grid.width);
            let shape_at = self
                .shapes_at
                .index(entry.shape as u32, self.shape_pos(entry.shape));
            let nodes_at = field!(shape_at, TileRoutingShapePod, wire_to_node);
            for (site, node_ref) in self.shape_nodes[entry.shape].sites(&nodes_at) {
                match node_ref.dx_mode {
                    RelNodeRefPod::MODE_TILE_WIRE => {}
                    RelNodeRefPod::MODE_IS_ROOT => self.check_root(&site, node_ref, xy)?,
                    _ => self.check_member(&site, node_ref, xy)?,
                }
            }
        }
        Ok(())
    }

    fn shape_pos(&self, shape: usize) -> usize {
        self.shapes.start + shape * size_of::<TileRoutingShapePod>()
    }

    fn check_root(
        &self,
        at: &Site<'_>,
        node_ref: &RelNodeRefPod,
        xy: (usize, usize),
    ) -> Result<(), ChipDbError> {
        let node = ChipDb::node_shape_idx(node_ref);
        let node = index(at, node.into(), self.node_shapes.len())?;
        let node_pos = self.node_shapes.start + node * size_of::<NodeShapePod>();
        let node_at = self.node_shapes_at.index(node as u32, node_pos);
        let members_at = field!(node_at, NodeShapePod, tile_wires);
        for (site, member) in self.node_members[node].sites(&members_at) {
            let dx_at = field!(site, RelTileWireRefPod, dx);
            let dy_at = field!(site, RelTileWireRefPod, dy);
            let target = self
                .grid
                .relative(xy, (&dx_at, &dy_at), member.dx, member.dy)?;
            let wire_at = field!(site, RelTileWireRefPod, wire);
            index(&wire_at, member.wire.into(), self.tiles[target].wires)?;
        }
        Ok(())
    }

    fn check_member(
        &self,
        at: &Site<'_>,
        node_ref: &RelNodeRefPod,
        xy: (usize, usize),
    ) -> Result<(), ChipDbError> {
        let dx_at = field!(at, RelNodeRefPod, dx_mode);
        let dy_at = field!(at, RelNodeRefPod, dy);
        let target = self
            .grid
            .relative(xy, (&dx_at, &dy_at), node_ref.dx_mode, node_ref.dy)?;
        let target = &self.tiles[target];
        let wire = index(
            &field!(at, RelNodeRefPod, wire),
            node_ref.wire.into(),
            target.wires,
        )?;
        let root = self.shape_nodes[target.shape].items.get(wire);
        match root.map(|root| root.dx_mode) {
            Some(RelNodeRefPod::MODE_IS_ROOT) => Ok(()),
            _ => Err(at.fault(Fault::NotNodeRoot)),
        }
    }
}

trait Walk: FromBytes + KnownLayout + Immutable {
    fn walk(&self, v: &Validator<'_>, at: &Site<'_>) -> Result<(), ChipDbError>;
}

macro_rules! walk {
    ($($pod:ty { $($field:ident: $kind:ident),* $(,)? })*) => {$(
        impl Walk for $pod {
            fn walk(&self, v: &Validator<'_>, at: &Site<'_>) -> Result<(), ChipDbError> {
                $(v.$kind(&field!(at, $pod, $field), self.$field)?;)*
                Ok(())
            }
        }
    )*};
}

walk! {
    ChipInfoPod {
        uarch: string,
        name: string,
        generator: opt_string,
        tile_types: nested,
        tile_insts: nested,
        node_shapes: nested,
        tile_shapes: nested,
        packages: nested,
        speed_grades: nested,
        extra_constids: opt_nested,
        extra_data: chip_extra,
    }
    ConstIdDataPod { bba_ids: strings }
    ChipExtraDataPod { packing_rules: flat }
    TileTypePod { bels: nested, wires: nested, pips: nested, groups: nested, extra_data: blob }
    BelDataPod { pins: flat, extra_data: blob }
    TileWireDataPod { pips_uphill: flat, pips_downhill: flat, bel_pins: flat }
    PipDataPod { extra_data: blob }
    GroupDataPod {
        group_bels: flat,
        group_wires: flat,
        group_pips: flat,
        group_groups: flat,
        extra_data: blob,
    }
    NodeShapePod { tile_wires: flat }
    TileRoutingShapePod { wire_to_node: flat }
    TileInstPod { extra_data: blob }
    PackageInfoPod { pads: nested, extra_data: blob }
    PadInfoPod { extra_data: blob }
    SpeedGradePod { pip_classes: flat, node_classes: flat, cell_types: nested, extra_data: blob }
    CellTimingPod { pins: nested }
    CellPinTimingPod { comb_arcs: flat, reg_arcs: flat }
}
