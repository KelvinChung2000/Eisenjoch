pods! {
    BelPinPod => BelPinPOD { name, wire, dir => type },
    BelDataPod => BelDataPOD { name, bel_type, z, padding, flags, site, checker_idx, pins, extra_data },
    BelPinRefPod => BelPinRefPOD { bel, pin },
    TileWireDataPod => TileWireDataPOD {
        name, wire_type, tile_wire, const_value, flags, timing_idx, pips_uphill, pips_downhill, bel_pins
    },
    PipDataPod => PipDataPOD { src_wire, dst_wire, pip_type => type, flags, timing_idx, extra_data },
    RelTileWireRefPod => RelTileWireRefPOD { dx, dy, wire },
    NodeShapePod => NodeShapePOD { tile_wires, timing_idx },
    GroupDataPod => GroupDataPOD {
        name, group_type, group_bels, group_wires, group_pips, group_groups, extra_data
    },
    TileTypePod => TileTypePOD { type_name, bels, wires, pips, groups, extra_data },
    RelNodeRefPod => RelNodeRefPOD { dx_mode, dy, wire },
    TileRoutingShapePod => TileRoutingShapePOD { wire_to_node, timing_index },
    PadInfoPod => PadInfoPOD { package_pin, tile, bel, pad_function, pad_bank, flags, extra_data },
    PackageInfoPod => PackageInfoPOD { name, pads, extra_data },
    TileInstPod => TileInstPOD { name_prefix, tile_type => type, shape, extra_data },
    TimingValue => TimingValue { fast_min, fast_max, slow_min, slow_max },
    PipTimingPod => PipTimingPOD { int_delay, in_cap, out_res, flags },
    NodeTimingPod => NodeTimingPOD { cap, res, delay },
    CellPinRegArcPod => CellPinRegArcPOD { clock, edge, setup, hold, clk_q },
    CellPinCombArcPod => CellPinCombArcPOD { input, delay },
    CellPinTimingPod => CellPinTimingPOD { pin, flags, comb_arcs, reg_arcs },
    CellTimingPod => CellTimingPOD { type_variant, pins },
    SpeedGradePod => SpeedGradePOD { name, pip_classes, node_classes, cell_types, extra_data },
    ConstIdDataPod => ConstIDDataPOD { known_id_count, bba_ids },
    ChipInfoPod => ChipInfoPOD {
        magic, version, width, height, uarch, name, generator, tile_types, tile_insts, node_shapes,
        tile_shapes, packages, speed_grades, extra_constids, extra_data
    },
}
