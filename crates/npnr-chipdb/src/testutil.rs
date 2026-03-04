//! Test utilities for creating synthetic chip databases.
//!
//! This module is only available when `feature = "test-utils"` or in `#[cfg(test)]`.
//! It provides helpers for building minimal in-memory chipdb binaries that can be
//! used in unit and integration tests without needing real FPGA chip database files.

use std::marker::PhantomData;
use std::mem;

use crate::pod::*;
use crate::relptr::{RelPtr, RelSlice};
use crate::CHIPDB_VERSION;

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

    /// Build a minimal chipdb with:
    /// - 2x2 grid (4 tiles)
    /// - 1 tile type with 1 bel ("LUT0" of type "LUT4", bucket "LUT"), 2 wires, 1 pip
    /// - Each tile is an instance of that tile type
    pub fn build_minimal() -> Vec<u8> {
        let mut db = SyntheticChipDbBuilder { buf: Vec::new() };

        // Reserve space for ChipInfoPod at the start
        let chip_info_offset = 0usize;
        let chip_info_size = mem::size_of::<ChipInfoPod>();
        db.buf.resize(chip_info_size, 0);

        // --- Strings ---
        let chip_name_offset = db.append_str("test_chip");
        let generator_offset = db.append_str("test_gen");
        let tile_type_name_offset = db.append_str("LOGIC");
        let bel_name_offset = db.append_str("LUT0");
        let bel_type_offset = db.append_str("LUT4");
        let bel_bucket_offset = db.append_str("LUT");
        let bel_pin_name_offset = db.append_str("I0");
        let wire0_name_offset = db.append_str("W0");
        let wire0_type_offset = db.append_str("LOCAL");
        let wire1_name_offset = db.append_str("W1");
        let wire1_type_offset = db.append_str("LOCAL");
        let tile_name_offsets = [
            db.append_str("TILE_0_0"),
            db.append_str("TILE_1_0"),
            db.append_str("TILE_0_1"),
            db.append_str("TILE_1_1"),
        ];

        // --- BelPinPod (1 pin) ---
        let bel_pin_offset = db.buf.len();
        let bel_pin_name_field_pos = bel_pin_offset;
        let bel_pin = BelPinPod {
            name: RelPtr {
                offset: Self::rel_offset(bel_pin_name_field_pos, bel_pin_name_offset),
                _phantom: PhantomData,
            },
            wire_index: 0,
            dir: 0, // PortType::In
        };
        db.append_val(&bel_pin);

        // --- BelDataPod (1 bel) ---
        let bel_data_offset = db.buf.len();
        let bel_name_field_pos = bel_data_offset;
        let bel_type_field_pos = bel_data_offset + 4;
        let bel_bucket_field_pos = bel_data_offset + 8;
        let bel_pins_offset_field_pos = bel_data_offset + 12;

        let bel = BelDataPod {
            name: RelPtr {
                offset: Self::rel_offset(bel_name_field_pos, bel_name_offset),
                _phantom: PhantomData,
            },
            bel_type: RelPtr {
                offset: Self::rel_offset(bel_type_field_pos, bel_type_offset),
                _phantom: PhantomData,
            },
            bucket: RelPtr {
                offset: Self::rel_offset(bel_bucket_field_pos, bel_bucket_offset),
                _phantom: PhantomData,
            },
            pins: RelSlice {
                offset: Self::rel_offset(bel_pins_offset_field_pos, bel_pin_offset),
                length: 1,
                _phantom: PhantomData,
            },
            extra_data: RelPtr {
                offset: 0,
                _phantom: PhantomData,
            },
            z: 0,
            padding: 0,
        };
        db.append_val(&bel);

        // --- PipDataPod (1 pip: wire 0 -> wire 1, in same tile) ---
        let pip_data_offset = db.buf.len();
        let pip = PipDataPod {
            src_wire: 0,
            dst_wire: 1,
            timing_index: -1,
            pip_type: 0,
            padding: 0,
            src_tile_delta: 0,
            dst_tile_delta: 0,
            extra_data: RelPtr {
                offset: 0,
                _phantom: PhantomData,
            },
        };
        db.append_val(&pip);

        // --- PipRefPod for wires (referencing the pip above) ---
        let pip_ref_offset = db.buf.len();
        let pip_ref = PipRefPod {
            tile_delta: 0,
            index: 0,
        };
        db.append_val(&pip_ref);

        // --- BelPinRefPod for wires ---
        let bel_pin_ref_offset = db.buf.len();
        let bel_pin_ref_pin_field = bel_pin_ref_offset + 4; // after bel(i32)
        let bel_pin_ref = BelPinRefPod {
            bel: 0,
            pin: RelPtr {
                offset: Self::rel_offset(bel_pin_ref_pin_field, bel_pin_name_offset),
                _phantom: PhantomData,
            },
        };
        db.append_val(&bel_pin_ref);

        // --- TileWireDataPod (2 wires) ---
        let wire0_offset = db.buf.len();
        let w0_name_field = wire0_offset;
        let w0_type_field = wire0_offset + 4;
        let w0_downhill_field = wire0_offset + 16;
        let w0_belpins_field = wire0_offset + 24;

        let wire0 = TileWireDataPod {
            name: RelPtr {
                offset: Self::rel_offset(w0_name_field, wire0_name_offset),
                _phantom: PhantomData,
            },
            wire_type: RelPtr {
                offset: Self::rel_offset(w0_type_field, wire0_type_offset),
                _phantom: PhantomData,
            },
            pips_uphill: RelSlice {
                offset: 0,
                length: 0,
                _phantom: PhantomData,
            },
            pips_downhill: RelSlice {
                offset: Self::rel_offset(w0_downhill_field, pip_ref_offset),
                length: 1,
                _phantom: PhantomData,
            },
            bel_pins: RelSlice {
                offset: Self::rel_offset(w0_belpins_field, bel_pin_ref_offset),
                length: 1,
                _phantom: PhantomData,
            },
            flags: 0,
        };
        db.append_val(&wire0);

        let wire1_offset = db.buf.len();
        let w1_name_field = wire1_offset;
        let w1_type_field = wire1_offset + 4;
        let w1_uphill_field = wire1_offset + 8;

        let wire1 = TileWireDataPod {
            name: RelPtr {
                offset: Self::rel_offset(w1_name_field, wire1_name_offset),
                _phantom: PhantomData,
            },
            wire_type: RelPtr {
                offset: Self::rel_offset(w1_type_field, wire1_type_offset),
                _phantom: PhantomData,
            },
            pips_uphill: RelSlice {
                offset: Self::rel_offset(w1_uphill_field, pip_ref_offset),
                length: 1,
                _phantom: PhantomData,
            },
            pips_downhill: RelSlice {
                offset: 0,
                length: 0,
                _phantom: PhantomData,
            },
            bel_pins: RelSlice {
                offset: 0,
                length: 0,
                _phantom: PhantomData,
            },
            flags: 0,
        };
        db.append_val(&wire1);

        // --- TileTypePod (1 tile type) ---
        let tile_type_offset = db.buf.len();
        let tt_name_field = tile_type_offset;
        let tt_bels_field = tile_type_offset + 4;
        let tt_wires_field = tile_type_offset + 12;
        let tt_pips_field = tile_type_offset + 20;

        let tile_type = TileTypePod {
            name: RelPtr {
                offset: Self::rel_offset(tt_name_field, tile_type_name_offset),
                _phantom: PhantomData,
            },
            bels: RelSlice {
                offset: Self::rel_offset(tt_bels_field, bel_data_offset),
                length: 1,
                _phantom: PhantomData,
            },
            wires: RelSlice {
                offset: Self::rel_offset(tt_wires_field, wire0_offset),
                length: 2,
                _phantom: PhantomData,
            },
            pips: RelSlice {
                offset: Self::rel_offset(tt_pips_field, pip_data_offset),
                length: 1,
                _phantom: PhantomData,
            },
            extra_data: RelPtr {
                offset: 0,
                _phantom: PhantomData,
            },
        };
        db.append_val(&tile_type);

        // --- TileInstPods (4 tiles in a 2x2 grid) ---
        let tile_insts_offset = db.buf.len();
        let coords: [(i16, i16); 4] = [(0, 0), (1, 0), (0, 1), (1, 1)];

        for (i, &(x, y)) in coords.iter().enumerate() {
            let inst_offset = db.buf.len();
            let inst_name_field = inst_offset;

            let inst = TileInstPod {
                name: RelPtr {
                    offset: Self::rel_offset(inst_name_field, tile_name_offsets[i]),
                    _phantom: PhantomData,
                },
                tile_type: 0, // all tiles use tile type 0
                tilewire_to_node: RelSlice {
                    offset: 0,
                    length: 0,
                    _phantom: PhantomData,
                },
                extra_data: RelPtr {
                    offset: 0,
                    _phantom: PhantomData,
                },
                x,
                y,
            };
            db.append_val(&inst);
        }

        // --- Fill in ChipInfoPod at offset 0 ---
        let ci_name_field = chip_info_offset + 16;
        let ci_gen_field = chip_info_offset + 20;
        let ci_tile_types_field = chip_info_offset + 24;
        let ci_tile_insts_field = chip_info_offset + 32;

        let chip_info = ChipInfoPod {
            version: CHIPDB_VERSION,
            width: 2,
            height: 2,
            num_tiles: 4,
            name: RelPtr {
                offset: Self::rel_offset(ci_name_field, chip_name_offset),
                _phantom: PhantomData,
            },
            generator: RelPtr {
                offset: Self::rel_offset(ci_gen_field, generator_offset),
                _phantom: PhantomData,
            },
            tile_types: RelSlice {
                offset: Self::rel_offset(ci_tile_types_field, tile_type_offset),
                length: 1,
                _phantom: PhantomData,
            },
            tile_insts: RelSlice {
                offset: Self::rel_offset(ci_tile_insts_field, tile_insts_offset),
                length: 4,
                _phantom: PhantomData,
            },
            nodes: RelSlice {
                offset: 0,
                length: 0,
                _phantom: PhantomData,
            },
            packages: RelSlice {
                offset: 0,
                length: 0,
                _phantom: PhantomData,
            },
            speed_grades: RelSlice {
                offset: 0,
                length: 0,
                _phantom: PhantomData,
            },
            extra_data: RelPtr {
                offset: 0,
                _phantom: PhantomData,
            },
        };

        db.write_at(chip_info_offset, &chip_info);
        db.buf
    }
}

/// Build a minimal synthetic ChipDb for testing.
///
/// Returns a `ChipDb` backed by a 2x2 grid with:
/// - 1 tile type ("LOGIC") shared by all tiles
/// - 1 bel per tile: "LUT0" of type "LUT4", bucket "LUT"
/// - 2 wires per tile: "W0" and "W1"
/// - 1 pip per tile: W0 -> W1
///
/// # Safety
/// This uses `from_bytes` internally which creates a temporary file for mmap.
pub fn make_test_chipdb() -> crate::ChipDb {
    let bytes = SyntheticChipDbBuilder::build_minimal();
    unsafe { crate::ChipDb::from_bytes(&bytes).expect("failed to load synthetic chipdb") }
}
