use super::*;
use crate::read_packed;
use crate::types::{BelId, Loc, PipId, WireId};
use std::ffi::CStr;

impl ChipDb {
    fn tile_type_checked(&self, tile: i32) -> Option<&TileTypePod> {
        let ci = self.chip_info();
        let tile_idx = usize::try_from(tile).ok()?;
        let inst = ci.tile_insts.get().get(tile_idx)?;
        let tt_idx: i32 = unsafe { read_packed!(*inst, tile_type) };
        let tt_usize = usize::try_from(tt_idx).ok()?;
        ci.tile_types.get().get(tt_usize)
    }

    fn tile_inst_checked(&self, tile: i32) -> Option<&TileInstPod> {
        let tile_idx = usize::try_from(tile).ok()?;
        self.chip_info().tile_insts.get().get(tile_idx)
    }

    fn bel_info_checked(&self, bel: BelId) -> Option<&BelDataPod> {
        let tt = self.tile_type_checked(bel.tile())?;
        let bel_idx = usize::try_from(bel.index()).ok()?;
        tt.bels.get().get(bel_idx)
    }

    fn wire_info_checked(&self, wire: WireId) -> Option<&TileWireDataPod> {
        let tt = self.tile_type_checked(wire.tile())?;
        let wire_idx = usize::try_from(wire.index()).ok()?;
        tt.wires.get().get(wire_idx)
    }

    fn pip_info_checked(&self, pip: PipId) -> Option<&PipDataPod> {
        let tt = self.tile_type_checked(pip.tile())?;
        let pip_idx = usize::try_from(pip.index()).ok()?;
        tt.pips.get().get(pip_idx)
    }

    #[inline]
    pub fn chip_info(&self) -> &ChipInfoPod {
        unsafe { &*self.chip_info }
    }

    pub fn constid_str(&self, index: i32) -> Option<&str> {
        if index < 0 || (index as usize) >= self.constid_strs.len() {
            return None;
        }
        self.constid_strs[index as usize].map(|ptr| unsafe {
            let cstr = CStr::from_ptr(ptr as *const std::ffi::c_char);
            cstr.to_str().unwrap_or("<invalid utf8>")
        })
    }

    #[inline]
    pub fn known_id_count(&self) -> i32 {
        self.known_id_count
    }

    pub fn extra_constids(&self) -> &[RelPtr<u8>] {
        let ci = self.chip_info();
        if ci.extra_constids.is_null() {
            return &[];
        }
        let data = unsafe { &*ci.extra_constids.get() };
        data.bba_ids.get()
    }

    #[inline]
    pub fn tile_type(&self, tile: i32) -> &TileTypePod {
        self.tile_type_checked(tile)
            .expect("tile_type: tile index out of bounds")
    }

    #[inline]
    pub fn tile_type_index(&self, tile: i32) -> i32 {
        let inst = self
            .tile_inst_checked(tile)
            .expect("tile_type_index: tile index out of bounds");
        unsafe { read_packed!(*inst, tile_type) }
    }

    #[inline]
    pub fn bel_info(&self, bel: BelId) -> &BelDataPod {
        self.bel_info_checked(bel)
            .expect("bel_info: BEL index out of bounds")
    }

    /// Extract (name_constid, wire_index) from a BEL pin, encapsulating unsafe access.
    #[inline]
    pub fn bel_pin_fields(&self, pin: &BelPinPod) -> (i32, i32) {
        let name: i32 = unsafe { read_packed!(*pin, name) };
        let wire: i32 = unsafe { read_packed!(*pin, wire) };
        (name, wire)
    }

    #[inline]
    pub fn wire_info(&self, wire: WireId) -> &TileWireDataPod {
        self.wire_info_checked(wire)
            .expect("wire_info: wire index out of bounds")
    }

    #[inline]
    pub fn pip_info(&self, pip: PipId) -> &PipDataPod {
        self.pip_info_checked(pip)
            .expect("pip_info: pip index out of bounds")
    }

    #[inline]
    pub fn tile_xy(&self, tile: i32) -> (i32, i32) {
        let w = self.width();
        (tile % w, tile / w)
    }

    #[inline]
    pub fn tile_by_xy(&self, x: i32, y: i32) -> i32 {
        y * self.width() + x
    }

    #[inline]
    pub fn rel_tile(&self, base: i32, dx: i32, dy: i32) -> i32 {
        let w = self.width();
        let x = base % w;
        let y = base / w;
        if dx == RelNodeRefPod::MODE_ROW_CONST as i32 {
            y * w
        } else if dx == RelNodeRefPod::MODE_GLB_CONST as i32 {
            0
        } else {
            (x + dx) + (y + dy) * w
        }
    }

    #[inline]
    pub fn width(&self) -> i32 {
        unsafe { read_packed!(*self.chip_info(), width) }
    }

    #[inline]
    pub fn height(&self) -> i32 {
        unsafe { read_packed!(*self.chip_info(), height) }
    }

    #[inline]
    pub fn num_tiles(&self) -> i32 {
        self.width() * self.height()
    }

    pub fn name(&self) -> &str {
        let ptr = self.chip_info().name.get();
        unsafe {
            let cstr = CStr::from_ptr(ptr as *const std::ffi::c_char);
            cstr.to_str().unwrap_or("<invalid utf8>")
        }
    }

    pub fn uarch(&self) -> &str {
        let ptr = self.chip_info().uarch.get();
        unsafe {
            let cstr = CStr::from_ptr(ptr as *const std::ffi::c_char);
            cstr.to_str().unwrap_or("<invalid utf8>")
        }
    }

    #[inline]
    pub fn tile_shape(&self, tile: i32) -> &TileRoutingShapePod {
        let ci = self.chip_info();
        let inst = &ci.tile_insts.get()[tile as usize];
        let shape_idx: i32 = unsafe { read_packed!(*inst, shape) };
        &ci.tile_shapes.get()[shape_idx as usize]
    }

    #[inline]
    pub fn node_shape(&self, index: u32) -> &NodeShapePod {
        &self.chip_info().node_shapes.get()[index as usize]
    }

    fn iter_tile_elements(
        &self,
        count_fn: fn(&TileTypePod) -> usize,
    ) -> impl Iterator<Item = (i32, usize)> + '_ {
        let ci = self.chip_info();
        let tile_insts = ci.tile_insts.get();
        let tile_types = ci.tile_types.get();

        tile_insts
            .iter()
            .enumerate()
            .map(move |(tile_idx, inst)| {
                let tt_idx: i32 = unsafe { read_packed!(*inst, tile_type) };
                let tt = &tile_types[tt_idx as usize];
                (tile_idx as i32, count_fn(tt))
            })
    }

    pub fn bels(&self) -> impl Iterator<Item = BelId> + '_ {
        self.iter_tile_elements(|tt| tt.bels.get().len())
            .flat_map(|(tile, count)| (0..count).map(move |i| BelId::new(tile, i as i32)))
    }

    pub fn wires(&self) -> impl Iterator<Item = WireId> + '_ {
        self.iter_tile_elements(|tt| tt.wires.get().len())
            .flat_map(|(tile, count)| (0..count).map(move |i| WireId::new(tile, i as i32)))
    }

    pub fn pips(&self) -> impl Iterator<Item = PipId> + '_ {
        self.iter_tile_elements(|tt| tt.pips.get().len())
            .flat_map(|(tile, count)| (0..count).map(move |i| PipId::new(tile, i as i32)))
    }

    pub fn bel_name(&self, bel: BelId) -> &str {
        let info = self.bel_info(bel);
        let name_id: i32 = unsafe { read_packed!(*info, name) };
        self.constid_str(name_id).unwrap_or("<unknown>")
    }

    pub fn bel_type(&self, bel: BelId) -> &str {
        let info = self.bel_info(bel);
        let type_id: i32 = unsafe { read_packed!(*info, bel_type) };
        self.constid_str(type_id).unwrap_or("<unknown>")
    }

    pub fn bel_loc(&self, bel: BelId) -> Loc {
        let (x, y) = self.tile_xy(bel.tile());
        let info = self.bel_info(bel);
        let z: i16 = unsafe { read_packed!(*info, z) };
        Loc::new(x, y, z as i32)
    }

    pub fn pip_src_wire(&self, pip: PipId) -> WireId {
        let info = self.pip_info(pip);
        let src_wire: i32 = unsafe { read_packed!(*info, src_wire) };
        WireId::new(pip.tile(), src_wire)
    }

    pub fn pip_dst_wire(&self, pip: PipId) -> WireId {
        let info = self.pip_info(pip);
        let dst_wire: i32 = unsafe { read_packed!(*info, dst_wire) };
        WireId::new(pip.tile(), dst_wire)
    }

    #[inline]
    pub fn tile_inst(&self, tile: i32) -> &TileInstPod {
        self.tile_inst_checked(tile)
            .expect("tile_inst: tile index out of bounds")
    }
}
