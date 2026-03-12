//! Test utilities for creating synthetic chip databases.
//!
//! This module is only available when `feature = "test-utils"` or in `#[cfg(test)]`.
//! It provides helpers for building minimal in-memory chipdb binaries that can be
//! used in unit and integration tests without needing real FPGA chip database files.
//!
//! The binary format starts with a `RelPtr<ChipInfoPod>` at offset 0, followed
//! by all data structures. String references are constid indices, resolved via
//! a `ConstIdDataPod` table.

use std::marker::PhantomData;
use std::mem;

use super::pod::*;
use super::relptr::{RelPtr, RelSlice};
use super::{CHIPDB_MAGIC, CHIPDB_VERSION};

/// Helper to build a synthetic chipdb binary blob in memory.
///
/// Constructs a valid minimal chipdb with configurable grid size, tile types,
/// bels, wires, and pips. All relative pointers are computed correctly.
pub struct SyntheticChipDbBuilder {
    buf: Vec<u8>,
}

impl SyntheticChipDbBuilder {
    /// Append a null-terminated string and return its offset.
    fn append_str(&mut self, s: &str) -> usize {
        let offset = self.buf.len();
        self.buf.extend_from_slice(s.as_bytes());
        self.buf.push(0); // null terminator
        offset
    }

    /// Append a value's raw bytes and return the offset.
    fn append_val<T>(&mut self, val: &T) -> usize {
        let offset = self.buf.len();
        let bytes = unsafe {
            std::slice::from_raw_parts(val as *const T as *const u8, mem::size_of::<T>())
        };
        self.buf.extend_from_slice(bytes);
        offset
    }

    /// Write a value at a specific offset.
    fn write_at<T>(&mut self, offset: usize, val: &T) {
        let bytes = unsafe {
            std::slice::from_raw_parts(val as *const T as *const u8, mem::size_of::<T>())
        };
        self.buf[offset..offset + bytes.len()].copy_from_slice(bytes);
    }

    /// Compute a relative offset: from `field_pos` to `target_pos`.
    fn rel_offset(field_pos: usize, target_pos: usize) -> i32 {
        (target_pos as isize - field_pos as isize) as i32
    }

    /// Make a RelPtr from field position to target position.
    fn make_relptr<T>(field_pos: usize, target_pos: usize) -> RelPtr<T> {
        RelPtr {
            offset: Self::rel_offset(field_pos, target_pos),
            _phantom: PhantomData,
        }
    }

    /// Make a RelSlice from field position to target position with given length.
    fn make_relslice<T>(field_pos: usize, target_pos: usize, length: u32) -> RelSlice<T> {
        RelSlice {
            offset: Self::rel_offset(field_pos, target_pos),
            length,
            _phantom: PhantomData,
        }
    }

    /// Make a null/empty RelPtr (offset 0).
    fn null_relptr<T>() -> RelPtr<T> {
        RelPtr {
            offset: 0,
            _phantom: PhantomData,
        }
    }

    /// Make an empty RelSlice (offset 0, length 0).
    fn empty_relslice<T>() -> RelSlice<T> {
        RelSlice {
            offset: 0,
            length: 0,
            _phantom: PhantomData,
        }
    }

    /// Pad the buffer to the given alignment.
    fn align(&mut self, alignment: usize) {
        let rem = self.buf.len() % alignment;
        if rem != 0 {
            self.buf.resize(self.buf.len() + (alignment - rem), 0);
        }
    }

    /// Create a `TimingValue` with symmetric min/max for fast and slow corners.
    fn tv(fast: i32, slow: i32) -> TimingValue {
        TimingValue {
            fast_min: fast, fast_max: fast,
            slow_min: slow, slow_max: slow,
        }
    }

    /// Build a minimal chipdb with:
    /// - 2x2 grid (4 tiles)
    /// - 1 tile type with 1 bel ("LUT0" of type "LUT4"), 2 wires, 1 pip
    /// - Each tile is an instance of that tile type
    /// - ConstIdData with all strings as bba_ids (known_id_count = 0)
    /// - Uniform timing: 1 speed grade ("DEFAULT") with:
    ///   - PIP timing class 0: 100/150ps delay (fast/slow)
    ///   - Node timing class 0: 50/75ps delay (fast/slow)
    ///   - Cell timing for LUT4: I0→O combinational arc, 200/300ps (fast/slow)
    pub fn build_minimal() -> Vec<u8> {
        let mut db = SyntheticChipDbBuilder { buf: Vec::new() };

        // =================================================================
        // Reserve space for root RelPtr<ChipInfoPod> at offset 0 (4 bytes)
        // =================================================================
        let root_relptr_offset = 0usize;
        db.buf.resize(mem::size_of::<RelPtr<ChipInfoPod>>(), 0);

        // =================================================================
        // Reserve space for ChipInfoPod right after root RelPtr
        // =================================================================
        let chip_info_offset = db.buf.len();
        let chip_info_size = mem::size_of::<ChipInfoPod>();
        db.buf.resize(chip_info_offset + chip_info_size, 0);

        // =================================================================
        // Strings (used as constid bba_ids entries)
        // =================================================================
        // Constid indices (known_id_count = 0, so index = position in bba_ids):
        //   0: "LOGIC"    (tile type name)
        //   1: "LUT0"     (bel name)
        //   2: "LUT4"     (bel type)
        //   3: "I0"       (bel input pin name)
        //   4: "W0"       (wire 0 name)
        //   5: "LOCAL"    (wire type)
        //   6: "W1"       (wire 1 name)
        //   7: "TILE_0_0" (tile name prefix)
        //   8: "TILE_1_0"
        //   9: "TILE_0_1"
        //  10: "TILE_1_1"
        //  11: "O"        (bel output pin name)
        //  12: "DEFAULT"  (speed grade name)
        let str_offsets = [
            db.append_str("LOGIC"),
            db.append_str("LUT0"),
            db.append_str("LUT4"),
            db.append_str("I0"),
            db.append_str("W0"),
            db.append_str("LOCAL"),
            db.append_str("W1"),
            db.append_str("TILE_0_0"),
            db.append_str("TILE_1_0"),
            db.append_str("TILE_0_1"),
            db.append_str("TILE_1_1"),
            db.append_str("O"),
            db.append_str("DEFAULT"),
        ];

        // Constid indices
        const ID_LOGIC: i32 = 0;
        const ID_LUT0: i32 = 1;
        const ID_LUT4: i32 = 2;
        const ID_I0: i32 = 3;
        const ID_W0: i32 = 4;
        const ID_LOCAL: i32 = 5;
        const ID_W1: i32 = 6;
        const ID_O: i32 = 11;
        const ID_DEFAULT: i32 = 12;

        // Direct string pointers for chip name/uarch/generator (not constids)
        let chip_name_offset = db.append_str("test_chip");
        let uarch_offset = db.append_str("test_uarch");
        let generator_offset = db.append_str("test_gen");

        // =================================================================
        // Build bba_ids array (RelPtr<u8> for each string)
        // =================================================================
        db.align(4); // RelPtr<u8> is 4 bytes, needs alignment
        let bba_ids_offset = db.buf.len();
        for &str_off in &str_offsets {
            let field_pos = db.buf.len();
            let relptr: RelPtr<u8> = Self::make_relptr(field_pos, str_off);
            db.append_val(&relptr);
        }

        // =================================================================
        // ConstIdDataPod
        // =================================================================
        let constid_data_offset = db.buf.len();
        let constid_bba_field = constid_data_offset + 4; // after known_id_count
        let constid_data = ConstIdDataPod {
            known_id_count: 0,
            bba_ids: Self::make_relslice(constid_bba_field, bba_ids_offset, str_offsets.len() as u32),
        };
        db.append_val(&constid_data);

        // =================================================================
        // BelPinPods (2 pins: "I0" input on wire 0, "O" output on wire 1)
        // =================================================================
        let bel_pin_offset = db.buf.len();
        let bel_pin0 = BelPinPod {
            name: ID_I0,
            wire: 0,
            dir: 0, // PortType::In
        };
        db.append_val(&bel_pin0);
        let bel_pin1 = BelPinPod {
            name: ID_O,
            wire: 1,
            dir: 1, // PortType::Out
        };
        db.append_val(&bel_pin1);

        // =================================================================
        // BelDataPod (1 bel: "LUT0" of type "LUT4", 2 pins)
        // =================================================================
        let bel_data_offset = db.buf.len();
        let bel_pins_field = bel_data_offset + 24; // name(4)+type(4)+z(2)+pad(2)+flags(4)+site(4)+checker_idx(4)=24

        let bel = BelDataPod {
            name: ID_LUT0,
            bel_type: ID_LUT4,
            z: 0,
            padding: 0,
            flags: 0,
            site: 0,
            checker_idx: 0,
            pins: Self::make_relslice(bel_pins_field, bel_pin_offset, 2),
            extra_data: Self::null_relptr(),
        };
        db.append_val(&bel);

        // =================================================================
        // PipDataPod (1 pip: wire 0 -> wire 1, timing class 0)
        // =================================================================
        let pip_data_offset = db.buf.len();
        let pip = PipDataPod {
            src_wire: 0,
            dst_wire: 1,
            pip_type: 0,
            flags: 0,
            timing_idx: 0,
            extra_data: Self::null_relptr(),
        };
        db.append_val(&pip);

        // =================================================================
        // Pip index arrays for wires (i32 indices, not PipRefPod)
        // =================================================================
        db.align(4); // i32 requires 4-byte alignment
        let pip_idx_offset = db.buf.len();
        let pip_idx: i32 = 0; // index of the pip in the tile's pip array
        db.append_val(&pip_idx);

        // =================================================================
        // BelPinRefPods for wires
        // =================================================================
        let bel_pin_ref0_offset = db.buf.len();
        let bel_pin_ref0 = BelPinRefPod {
            bel: 0,
            pin: ID_I0,
        };
        db.append_val(&bel_pin_ref0);

        let bel_pin_ref1_offset = db.buf.len();
        let bel_pin_ref1 = BelPinRefPod {
            bel: 0,
            pin: ID_O,
        };
        db.append_val(&bel_pin_ref1);

        // =================================================================
        // TileWireDataPod (2 wires: W0 and W1)
        // =================================================================
        let wire0_offset = db.buf.len();
        // Wire 0: downhill pip [0], bel pin ref (I0), no uphill
        // Field offsets: name(4)+type(4)+tile_wire(4)+const(4)+flags(4)+timing(4)=24, then relslices of 8 each
        let w0_pips_down_field = wire0_offset + 32;
        let w0_belpins_field = wire0_offset + 40;

        let wire0 = TileWireDataPod {
            name: ID_W0,
            wire_type: ID_LOCAL,
            tile_wire: 0,
            const_value: 0,
            flags: 0,
            timing_idx: 0,
            pips_uphill: Self::empty_relslice(),
            pips_downhill: Self::make_relslice(w0_pips_down_field, pip_idx_offset, 1),
            bel_pins: Self::make_relslice(w0_belpins_field, bel_pin_ref0_offset, 1),
        };
        db.append_val(&wire0);

        let wire1_offset = db.buf.len();
        // Wire 1: uphill pip [0], bel pin ref (O), no downhill
        let w1_pips_up_field = wire1_offset + 24;
        let w1_belpins_field = wire1_offset + 40;

        let wire1 = TileWireDataPod {
            name: ID_W1,
            wire_type: ID_LOCAL,
            tile_wire: 1,
            const_value: 0,
            flags: 0,
            timing_idx: 0,
            pips_uphill: Self::make_relslice(w1_pips_up_field, pip_idx_offset, 1),
            pips_downhill: Self::empty_relslice(),
            bel_pins: Self::make_relslice(w1_belpins_field, bel_pin_ref1_offset, 1),
        };
        db.append_val(&wire1);

        // =================================================================
        // TileTypePod (1 tile type: "LOGIC")
        // =================================================================
        let tile_type_offset = db.buf.len();
        let tt_bels_field = tile_type_offset + 4;      // after type_name(4)
        let tt_wires_field = tile_type_offset + 12;     // +8
        let tt_pips_field = tile_type_offset + 20;      // +8

        let tile_type = TileTypePod {
            type_name: ID_LOGIC,
            bels: Self::make_relslice(tt_bels_field, bel_data_offset, 1),
            wires: Self::make_relslice(tt_wires_field, wire0_offset, 2),
            pips: Self::make_relslice(tt_pips_field, pip_data_offset, 1),
            groups: Self::empty_relslice(),
            extra_data: Self::null_relptr(),
        };
        db.append_val(&tile_type);

        // =================================================================
        // TileRoutingShapePod (1 shape, timing class 0)
        // =================================================================
        let tile_shape_offset = db.buf.len();
        let tile_routing_shape = TileRoutingShapePod {
            wire_to_node: Self::empty_relslice(),
            timing_index: 0,
        };
        db.append_val(&tile_routing_shape);

        // =================================================================
        // TileInstPods (4 tiles in a 2x2 grid)
        // =================================================================
        let tile_insts_offset = db.buf.len();
        let tile_name_constids: [i32; 4] = [7, 8, 9, 10]; // TILE_0_0, TILE_1_0, TILE_0_1, TILE_1_1

        for &name_id in &tile_name_constids {
            let inst = TileInstPod {
                name_prefix: name_id,
                tile_type: 0, // all tiles use tile type 0
                shape: 0,     // all tiles use shape 0
                extra_data: Self::null_relptr(),
            };
            db.append_val(&inst);
        }

        // =================================================================
        // Timing data: uniform delays for all pips, nodes, and cell arcs
        // =================================================================
        let tv_zero = Self::tv(0, 0);
        let tv_pip_delay = Self::tv(100, 150);
        let tv_node_delay = Self::tv(50, 75);
        let tv_cell_delay = Self::tv(200, 300);

        // PipTimingPod (class 0): 100/150ps delay
        let pip_timing_offset = db.buf.len();
        let pip_timing = PipTimingPod {
            int_delay: tv_pip_delay,
            in_cap: tv_zero,
            out_res: tv_zero,
            flags: 0,
        };
        db.append_val(&pip_timing);

        // NodeTimingPod (class 0): 50/75ps delay
        let node_timing_offset = db.buf.len();
        let node_timing = NodeTimingPod {
            cap: tv_zero,
            res: tv_zero,
            delay: tv_node_delay,
        };
        db.append_val(&node_timing);

        // CellPinCombArcPod: I0 → O, 200/300ps
        let comb_arc_offset = db.buf.len();
        let comb_arc = CellPinCombArcPod {
            input: ID_I0,
            delay: tv_cell_delay,
        };
        db.append_val(&comb_arc);

        // CellPinTimingPod: output pin "O" with one combinational arc from I0
        let cell_pin_timing_offset = db.buf.len();
        // Fields: pin(4) + flags(4) + comb_arcs(8) + reg_arcs(8) = 24
        let cpt_comb_field = cell_pin_timing_offset + 8;
        let cell_pin_timing = CellPinTimingPod {
            pin: ID_O,
            flags: 0,
            comb_arcs: Self::make_relslice(cpt_comb_field, comb_arc_offset, 1),
            reg_arcs: Self::empty_relslice(),
        };
        db.append_val(&cell_pin_timing);

        // CellTimingPod: LUT4 cell type
        let cell_timing_offset = db.buf.len();
        // Fields: type_variant(4) + pins(8) = 12
        let ct_pins_field = cell_timing_offset + 4;
        let cell_timing = CellTimingPod {
            type_variant: ID_LUT4,
            pins: Self::make_relslice(ct_pins_field, cell_pin_timing_offset, 1),
        };
        db.append_val(&cell_timing);

        // SpeedGradePod: "DEFAULT" speed grade with all timing classes
        let speed_grade_offset = db.buf.len();
        // Fields: name(4) + pip_classes(8) + node_classes(8) + cell_types(8) = 28
        let sg_pip_field = speed_grade_offset + 4;
        let sg_node_field = speed_grade_offset + 12;
        let sg_cell_field = speed_grade_offset + 20;
        let speed_grade = SpeedGradePod {
            name: ID_DEFAULT,
            pip_classes: Self::make_relslice(sg_pip_field, pip_timing_offset, 1),
            node_classes: Self::make_relslice(sg_node_field, node_timing_offset, 1),
            cell_types: Self::make_relslice(sg_cell_field, cell_timing_offset, 1),
        };
        db.append_val(&speed_grade);

        // =================================================================
        // Fill in ChipInfoPod
        // =================================================================
        let ci = chip_info_offset;
        let ci_uarch_field = ci + 16;           // magic(4)+version(4)+width(4)+height(4)
        let ci_name_field = ci + 20;
        let ci_gen_field = ci + 24;
        let ci_tile_types_field = ci + 28;
        let ci_tile_insts_field = ci + 36;      // +8 (RelSlice)
        let ci_tile_shapes_field = ci + 52;     // +8+8 (skip node_shapes)
        let ci_speed_grades_field = ci + 68;    // +8+8 (skip packages)
        let ci_extra_constids_field = ci + 76;  // +8 (skip speed_grades)

        let chip_info = ChipInfoPod {
            magic: CHIPDB_MAGIC,
            version: CHIPDB_VERSION,
            width: 2,
            height: 2,
            uarch: Self::make_relptr(ci_uarch_field, uarch_offset),
            name: Self::make_relptr(ci_name_field, chip_name_offset),
            generator: Self::make_relptr(ci_gen_field, generator_offset),
            tile_types: Self::make_relslice(ci_tile_types_field, tile_type_offset, 1),
            tile_insts: Self::make_relslice(ci_tile_insts_field, tile_insts_offset, 4),
            node_shapes: Self::empty_relslice(),
            tile_shapes: Self::make_relslice(ci_tile_shapes_field, tile_shape_offset, 1),
            packages: Self::empty_relslice(),
            speed_grades: Self::make_relslice(ci_speed_grades_field, speed_grade_offset, 1),
            extra_constids: Self::make_relptr(ci_extra_constids_field, constid_data_offset),
            extra_data: Self::null_relptr(),
        };
        db.write_at(chip_info_offset, &chip_info);

        // =================================================================
        // Fill in root RelPtr at offset 0
        // =================================================================
        let root_relptr: RelPtr<ChipInfoPod> = Self::make_relptr(root_relptr_offset, chip_info_offset);
        db.write_at(root_relptr_offset, &root_relptr);

        db.buf
    }
}

/// Build a minimal synthetic ChipDb for testing.
///
/// Returns a `ChipDb` backed by a 2x2 grid with:
/// - 1 tile type ("LOGIC") shared by all tiles
/// - 1 bel per tile: "LUT0" of type "LUT4" with pins I0 (input) and O (output)
/// - 2 wires per tile: "W0" (connected to I0) and "W1" (connected to O)
/// - 1 pip per tile: W0 -> W1
/// - 1 speed grade ("DEFAULT") with uniform timing:
///   - PIP delay: 100ps fast / 150ps slow
///   - Node delay: 50ps fast / 75ps slow
///   - LUT4 I0→O: 200ps fast / 300ps slow
///
/// # Safety
/// This uses `from_bytes` internally which creates a temporary file for mmap.
pub fn make_test_chipdb() -> super::ChipDb {
    let bytes = SyntheticChipDbBuilder::build_minimal();
    unsafe {
        match super::ChipDb::from_bytes(&bytes) {
            Ok(chipdb) => chipdb,
            Err(err) => panic!("failed to load synthetic chipdb: {err}"),
        }
    }
}
