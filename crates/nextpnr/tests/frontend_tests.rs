use nextpnr::frontend::*;
use nextpnr::types::*;

fn make_pool() -> IdStringPool {
    IdStringPool::new()
}

// =========================================================================
// Helper unit tests
// =========================================================================

#[test]
fn test_parse_bit_value_signal() {
    let val = serde_json::json!(42);
    assert_eq!(parse_bit_value(&val).unwrap(), BitValue::Signal(42));
}

#[test]
fn test_parse_bit_value_zero() {
    let val = serde_json::json!("0");
    assert_eq!(parse_bit_value(&val).unwrap(), BitValue::Zero);
}

#[test]
fn test_parse_bit_value_one() {
    let val = serde_json::json!("1");
    assert_eq!(parse_bit_value(&val).unwrap(), BitValue::One);
}

#[test]
fn test_parse_bit_value_x() {
    let val = serde_json::json!("x");
    assert_eq!(parse_bit_value(&val).unwrap(), BitValue::Undef);
}

#[test]
fn test_parse_bit_value_z() {
    let val = serde_json::json!("z");
    assert_eq!(parse_bit_value(&val).unwrap(), BitValue::Undef);
}

#[test]
fn test_parse_bit_value_invalid_string() {
    let val = serde_json::json!("abc");
    assert!(parse_bit_value(&val).is_err());
}

#[test]
fn test_parse_bit_value_invalid_type() {
    let val = serde_json::json!(true);
    assert!(parse_bit_value(&val).is_err());
}

#[test]
fn test_parse_port_direction() {
    assert_eq!(parse_port_direction("input").unwrap(), PortType::In);
    assert_eq!(parse_port_direction("output").unwrap(), PortType::Out);
    assert_eq!(parse_port_direction("inout").unwrap(), PortType::InOut);
    assert!(parse_port_direction("unknown").is_err());
}

#[test]
fn test_parse_property_integer() {
    let val = serde_json::json!(42);
    let prop = parse_property(&val).unwrap();
    assert_eq!(prop, Property::int(42));
}

#[test]
fn test_parse_property_binary_string() {
    let val = serde_json::json!("0000000000001111");
    let prop = parse_property(&val).unwrap();
    assert_eq!(prop, Property::bit_vector("0000000000001111"));
    assert_eq!(prop.as_int(), Some(15));
}

#[test]
fn test_parse_property_plain_string() {
    let val = serde_json::json!("blinky.v:5.3-5.20");
    let prop = parse_property(&val).unwrap();
    assert_eq!(prop, Property::string("blinky.v:5.3-5.20"));
}

#[test]
fn test_port_bit_name_single() {
    assert_eq!(port_bit_name("clk", 0, 1), "clk");
}

#[test]
fn test_port_bit_name_multi() {
    assert_eq!(port_bit_name("data", 0, 4), "data[0]");
    assert_eq!(port_bit_name("data", 3, 4), "data[3]");
}

// =========================================================================
// Integration tests: minimal blinky design
// =========================================================================

const BLINKY_JSON: &str = r#"{
    "creator": "Yosys 0.9+4081",
    "modules": {
        "top": {
            "attributes": {
                "top": "00000000000000000000000000000001"
            },
            "parameter_default_values": {},
            "ports": {
                "clk": { "direction": "input", "bits": [2] },
                "led": { "direction": "output", "bits": [3] }
            },
            "cells": {
                "lut0": {
                    "hide_name": 0,
                    "type": "LUT4",
                    "parameters": {
                        "INIT": "0000000000001111"
                    },
                    "attributes": {
                        "src": "blinky.v:5.3-5.20"
                    },
                    "port_directions": {
                        "A": "input",
                        "B": "input",
                        "C": "input",
                        "D": "input",
                        "Z": "output"
                    },
                    "connections": {
                        "A": [2],
                        "B": ["0"],
                        "C": ["0"],
                        "D": ["0"],
                        "Z": [3]
                    }
                }
            },
            "netnames": {
                "clk": {
                    "hide_name": 0,
                    "bits": [2],
                    "attributes": {}
                },
                "led": {
                    "hide_name": 0,
                    "bits": [3],
                    "attributes": {}
                }
            }
        }
    }
}"#;

#[test]
fn test_blinky_parse_succeeds() {
    let pool = make_pool();
    let design = parse_json(BLINKY_JSON, &pool).unwrap();
    assert!(!design.top_module.is_empty());
    assert_eq!(pool.lookup(design.top_module), Some("top"));
}

#[test]
fn test_blinky_has_cells() {
    let pool = make_pool();
    let design = parse_json(BLINKY_JSON, &pool).unwrap();

    // Should have: lut0, $PACKER_GND, $PACKER_VCC, $io$clk, $io$led
    let lut0_id = pool.intern("lut0");
    assert!(design.cell_by_name(lut0_id).is_some());

    let gnd_id = pool.intern("$PACKER_GND");
    assert!(design.cell_by_name(gnd_id).is_some());

    let vcc_id = pool.intern("$PACKER_VCC");
    assert!(design.cell_by_name(vcc_id).is_some());

    let io_clk_id = pool.intern("$io$clk");
    assert!(design.cell_by_name(io_clk_id).is_some());

    let io_led_id = pool.intern("$io$led");
    assert!(design.cell_by_name(io_led_id).is_some());
}

#[test]
fn test_blinky_lut_type() {
    let pool = make_pool();
    let design = parse_json(BLINKY_JSON, &pool).unwrap();

    let lut0_id = pool.intern("lut0");
    let lut4_type = pool.intern("LUT4");
    let cell_idx = design.cell_by_name(lut0_id).unwrap();
    let cell = design.cell(cell_idx);
    assert_eq!(cell.cell_type, lut4_type);
}

#[test]
fn test_blinky_lut_ports() {
    let pool = make_pool();
    let design = parse_json(BLINKY_JSON, &pool).unwrap();

    let lut0_id = pool.intern("lut0");
    let cell_idx = design.cell_by_name(lut0_id).unwrap();
    let cell = design.cell(cell_idx);

    // Check port A exists and is input
    let port_a = pool.intern("A");
    let pa = cell.port(port_a).unwrap();
    assert_eq!(pa.port_type, PortType::In);
    assert!(pa.net.is_some()); // connected to signal net 2

    // Check port Z exists and is output
    let port_z = pool.intern("Z");
    let pz = cell.port(port_z).unwrap();
    assert_eq!(pz.port_type, PortType::Out);
    assert!(pz.net.is_some()); // connected to signal net 3
}

#[test]
fn test_blinky_lut_parameters() {
    let pool = make_pool();
    let design = parse_json(BLINKY_JSON, &pool).unwrap();

    let lut0_id = pool.intern("lut0");
    let cell_idx = design.cell_by_name(lut0_id).unwrap();
    let cell = design.cell(cell_idx);

    let init_key = pool.intern("INIT");
    let init = cell.params.get(&init_key).unwrap();
    assert_eq!(init.as_int(), Some(15)); // 0000000000001111 in binary = 15
}

#[test]
fn test_blinky_lut_attributes() {
    let pool = make_pool();
    let design = parse_json(BLINKY_JSON, &pool).unwrap();

    let lut0_id = pool.intern("lut0");
    let cell_idx = design.cell_by_name(lut0_id).unwrap();
    let cell = design.cell(cell_idx);

    let src_key = pool.intern("src");
    let src = cell.attrs.get(&src_key).unwrap();
    assert_eq!(src.as_str(), "blinky.v:5.3-5.20");
}

#[test]
fn test_blinky_constant_connections() {
    let pool = make_pool();
    let design = parse_json(BLINKY_JSON, &pool).unwrap();

    // Ports B, C, D of lut0 are connected to "0" -> GND net
    let lut0_id = pool.intern("lut0");
    let cell_idx = design.cell_by_name(lut0_id).unwrap();
    let cell = design.cell(cell_idx);

    let gnd_net_id = pool.intern("$PACKER_GND_NET");
    let gnd_net_idx = design.net_by_name(gnd_net_id).unwrap();

    let port_b = pool.intern("B");
    let pb = cell.port(port_b).unwrap();
    assert_eq!(pb.net, Some(gnd_net_idx));

    let port_c = pool.intern("C");
    let pc = cell.port(port_c).unwrap();
    assert_eq!(pc.net, Some(gnd_net_idx));

    let port_d = pool.intern("D");
    let pd = cell.port(port_d).unwrap();
    assert_eq!(pd.net, Some(gnd_net_idx));
}

#[test]
fn test_blinky_has_nets() {
    let pool = make_pool();
    let design = parse_json(BLINKY_JSON, &pool).unwrap();

    // Should have nets: clk (renamed from $signal$2), led (renamed from $signal$3),
    // $PACKER_GND_NET, $PACKER_VCC_NET
    let gnd_net_id = pool.intern("$PACKER_GND_NET");
    assert!(design.net_by_name(gnd_net_id).is_some());

    let vcc_net_id = pool.intern("$PACKER_VCC_NET");
    assert!(design.net_by_name(vcc_net_id).is_some());

    // The signal nets get renamed by netnames
    let clk_net = pool.intern("clk");
    assert!(design.net_by_name(clk_net).is_some());

    let led_net = pool.intern("led");
    assert!(design.net_by_name(led_net).is_some());
}

#[test]
fn test_blinky_io_cells() {
    let pool = make_pool();
    let design = parse_json(BLINKY_JSON, &pool).unwrap();

    // clk is an input -> $nextpnr_IBUF pseudo-cell
    let io_clk_id = pool.intern("$io$clk");
    let ibuf_type = pool.intern("$nextpnr_IBUF");
    let clk_cell_idx = design.cell_by_name(io_clk_id).unwrap();
    let clk_cell = design.cell(clk_cell_idx);
    assert_eq!(clk_cell.cell_type, ibuf_type);

    // led is an output -> $nextpnr_OBUF pseudo-cell
    let io_led_id = pool.intern("$io$led");
    let obuf_type = pool.intern("$nextpnr_OBUF");
    let led_cell_idx = design.cell_by_name(io_led_id).unwrap();
    let led_cell = design.cell(led_cell_idx);
    assert_eq!(led_cell.cell_type, obuf_type);
}

#[test]
fn test_blinky_net_connectivity() {
    let pool = make_pool();
    let design = parse_json(BLINKY_JSON, &pool).unwrap();

    // The clk net (signal 2) should be:
    //  - driven by the IBUF (output port O)
    //  - used by lut0 port A
    let clk_net_id = pool.intern("clk");
    let clk_net_idx = design.net_by_name(clk_net_id).unwrap();
    let clk_net = design.net(clk_net_idx);

    // The IBUF's O port drives this net
    assert!(clk_net.has_driver());
    let io_clk_id = pool.intern("$io$clk");
    let io_clk_idx = design.cell_by_name(io_clk_id).unwrap();
    assert_eq!(clk_net.driver.cell, Some(io_clk_idx));

    // lut0's A port is a user
    assert!(clk_net.num_users() >= 1);
    let lut0_id = pool.intern("lut0");
    let lut0_idx = design.cell_by_name(lut0_id).unwrap();
    let has_lut_user = clk_net
        .users
        .iter()
        .any(|u| u.cell == Some(lut0_idx) && u.port == pool.intern("A"));
    assert!(has_lut_user);
}

// =========================================================================
// Multi-bit port test
// =========================================================================

const MULTIBIT_JSON: &str = r#"{
    "creator": "Yosys",
    "modules": {
        "multi": {
            "attributes": { "top": "00000000000000000000000000000001" },
            "parameter_default_values": {},
            "ports": {
                "data_in": { "direction": "input", "bits": [10, 11, 12, 13] },
                "data_out": { "direction": "output", "bits": [20, 21, 22, 23] }
            },
            "cells": {
                "buf0": {
                    "hide_name": 0,
                    "type": "BUF4",
                    "parameters": {},
                    "attributes": {},
                    "port_directions": {
                        "I": "input",
                        "O": "output"
                    },
                    "connections": {
                        "I": [10, 11, 12, 13],
                        "O": [20, 21, 22, 23]
                    }
                }
            },
            "netnames": {}
        }
    }
}"#;

#[test]
fn test_multibit_port_names() {
    let pool = make_pool();
    let design = parse_json(MULTIBIT_JSON, &pool).unwrap();

    let buf0_id = pool.intern("buf0");
    let cell_idx = design.cell_by_name(buf0_id).unwrap();
    let cell = design.cell(cell_idx);

    // Multi-bit port I should become I[0], I[1], I[2], I[3]
    for i in 0..4 {
        let pname = pool.intern(&format!("I[{}]", i));
        assert!(cell.port(pname).is_some(), "Missing port I[{}]", i);
        assert_eq!(cell.port(pname).unwrap().port_type, PortType::In);
    }

    // Multi-bit port O should become O[0], O[1], O[2], O[3]
    for i in 0..4 {
        let pname = pool.intern(&format!("O[{}]", i));
        assert!(cell.port(pname).is_some(), "Missing port O[{}]", i);
        assert_eq!(cell.port(pname).unwrap().port_type, PortType::Out);
    }
}

#[test]
fn test_multibit_io_cells() {
    let pool = make_pool();
    let design = parse_json(MULTIBIT_JSON, &pool).unwrap();

    // Multi-bit input port data_in -> 4 IBUF cells
    for i in 0..4 {
        let cell_name = pool.intern(&format!("$io$data_in[{}]", i));
        assert!(
            design.cell_by_name(cell_name).is_some(),
            "Missing IO cell for data_in[{}]",
            i
        );
        let cell_idx = design.cell_by_name(cell_name).unwrap();
        let cell = design.cell(cell_idx);
        assert_eq!(cell.cell_type, pool.intern("$nextpnr_IBUF"));
    }

    // Multi-bit output port data_out -> 4 OBUF cells
    for i in 0..4 {
        let cell_name = pool.intern(&format!("$io$data_out[{}]", i));
        assert!(
            design.cell_by_name(cell_name).is_some(),
            "Missing IO cell for data_out[{}]",
            i
        );
        let cell_idx = design.cell_by_name(cell_name).unwrap();
        let cell = design.cell(cell_idx);
        assert_eq!(cell.cell_type, pool.intern("$nextpnr_OBUF"));
    }
}

#[test]
fn test_multibit_net_connections() {
    let pool = make_pool();
    let design = parse_json(MULTIBIT_JSON, &pool).unwrap();

    let buf0_id = pool.intern("buf0");
    let buf0_idx = design.cell_by_name(buf0_id).unwrap();

    // Each input bit of buf0 should be connected to the corresponding IBUF output
    for i in 0..4 {
        let port_name = pool.intern(&format!("I[{}]", i));
        let port = design.cell(buf0_idx).port(port_name).unwrap();
        assert!(port.net.is_some(), "buf0 I[{}] should be connected", i);
    }

    // Each output bit of buf0 should drive a net
    for i in 0..4 {
        let port_name = pool.intern(&format!("O[{}]", i));
        let port = design.cell(buf0_idx).port(port_name).unwrap();
        assert!(port.net.is_some(), "buf0 O[{}] should be connected", i);
    }
}

// =========================================================================
// Constant driver bits test
// =========================================================================

const CONSTANTS_JSON: &str = r#"{
    "creator": "Yosys",
    "modules": {
        "consts": {
            "attributes": { "top": "00000000000000000000000000000001" },
            "parameter_default_values": {},
            "ports": {
                "out_a": { "direction": "output", "bits": [5] },
                "out_b": { "direction": "output", "bits": [6] }
            },
            "cells": {
                "cell_a": {
                    "hide_name": 0,
                    "type": "BUF",
                    "parameters": {},
                    "attributes": {},
                    "port_directions": { "I": "input", "O": "output" },
                    "connections": { "I": ["1"], "O": [5] }
                },
                "cell_b": {
                    "hide_name": 0,
                    "type": "BUF",
                    "parameters": {},
                    "attributes": {},
                    "port_directions": { "I": "input", "O": "output" },
                    "connections": { "I": ["x"], "O": [6] }
                }
            },
            "netnames": {}
        }
    }
}"#;

#[test]
fn test_constant_one_connection() {
    let pool = make_pool();
    let design = parse_json(CONSTANTS_JSON, &pool).unwrap();

    let cell_a_id = pool.intern("cell_a");
    let cell_idx = design.cell_by_name(cell_a_id).unwrap();
    let cell = design.cell(cell_idx);

    let port_i = pool.intern("I");
    let pi = cell.port(port_i).unwrap();

    // Should be connected to VCC net
    let vcc_net_id = pool.intern("$PACKER_VCC_NET");
    let vcc_net_idx = design.net_by_name(vcc_net_id).unwrap();
    assert_eq!(pi.net, Some(vcc_net_idx));
}

#[test]
fn test_constant_x_unconnected() {
    let pool = make_pool();
    let design = parse_json(CONSTANTS_JSON, &pool).unwrap();

    let cell_b_id = pool.intern("cell_b");
    let cell_idx = design.cell_by_name(cell_b_id).unwrap();
    let cell = design.cell(cell_idx);

    // "x" means unconnected
    let port_i = pool.intern("I");
    let pi = cell.port(port_i).unwrap();
    assert!(pi.net.is_none());
}

// =========================================================================
// Empty module test
// =========================================================================

#[test]
fn test_empty_module() {
    let json = r#"{
        "creator": "Yosys",
        "modules": {
            "empty": {
                "attributes": {},
                "parameter_default_values": {},
                "ports": {},
                "cells": {},
                "netnames": {}
            }
        }
    }"#;
    let pool = make_pool();
    let design = parse_json(json, &pool).unwrap();

    // Should have only the GND/VCC cells and nets
    assert_eq!(design.num_cells(), 2); // GND + VCC
    assert_eq!(design.num_nets(), 2); // GND_NET + VCC_NET
}

// =========================================================================
// No modules error test
// =========================================================================

#[test]
fn test_no_modules_error() {
    let json = r#"{ "creator": "Yosys" }"#;
    let pool = make_pool();
    assert!(parse_json(json, &pool).is_err());
}

#[test]
fn test_empty_modules_error() {
    let json = r#"{ "modules": {} }"#;
    let pool = make_pool();
    assert!(parse_json(json, &pool).is_err());
}

#[test]
fn test_invalid_json_error() {
    let pool = make_pool();
    assert!(parse_json("not json at all", &pool).is_err());
}

// =========================================================================
// Top module selection test
// =========================================================================

#[test]
fn test_top_module_selection_by_attribute() {
    let json = r#"{
        "modules": {
            "sub": {
                "attributes": {},
                "parameter_default_values": {},
                "ports": {},
                "cells": {},
                "netnames": {}
            },
            "main": {
                "attributes": { "top": "00000000000000000000000000000001" },
                "parameter_default_values": {},
                "ports": {},
                "cells": {},
                "netnames": {}
            }
        }
    }"#;
    let pool = make_pool();
    let design = parse_json(json, &pool).unwrap();
    assert_eq!(pool.lookup(design.top_module), Some("main"));
}

// =========================================================================
// Inout port test
// =========================================================================

#[test]
fn test_inout_port() {
    let json = r#"{
        "modules": {
            "bidir": {
                "attributes": { "top": "00000000000000000000000000000001" },
                "parameter_default_values": {},
                "ports": {
                    "sda": { "direction": "inout", "bits": [2] }
                },
                "cells": {},
                "netnames": {}
            }
        }
    }"#;
    let pool = make_pool();
    let design = parse_json(json, &pool).unwrap();

    let io_sda = pool.intern("$io$sda");
    let cell_idx = design.cell_by_name(io_sda).unwrap();
    let cell = design.cell(cell_idx);
    assert_eq!(cell.cell_type, pool.intern("$nextpnr_IOBUF"));
}

// =========================================================================
// GND net users test
// =========================================================================

#[test]
fn test_gnd_net_has_users() {
    let pool = make_pool();
    let design = parse_json(BLINKY_JSON, &pool).unwrap();

    let gnd_net_id = pool.intern("$PACKER_GND_NET");
    let gnd_net_idx = design.net_by_name(gnd_net_id).unwrap();
    let gnd_net = design.net(gnd_net_idx);

    // GND net should have users: lut0 ports B, C, D
    assert_eq!(gnd_net.num_users(), 3);
}

#[test]
fn test_gnd_net_has_driver() {
    let pool = make_pool();
    let design = parse_json(BLINKY_JSON, &pool).unwrap();

    let gnd_net_id = pool.intern("$PACKER_GND_NET");
    let gnd_net_idx = design.net_by_name(gnd_net_id).unwrap();
    let gnd_net = design.net(gnd_net_idx);

    assert!(gnd_net.has_driver());
    let gnd_cell_id = pool.intern("$PACKER_GND");
    let gnd_cell_idx = design.cell_by_name(gnd_cell_id).unwrap();
    assert_eq!(gnd_net.driver.cell, Some(gnd_cell_idx));
}

// =========================================================================
// Multiple parameters test
// =========================================================================

#[test]
fn test_multiple_parameters() {
    let json = r#"{
        "modules": {
            "top": {
                "attributes": { "top": "00000000000000000000000000000001" },
                "parameter_default_values": {},
                "ports": {},
                "cells": {
                    "dff0": {
                        "hide_name": 0,
                        "type": "FDRE",
                        "parameters": {
                            "INIT": "0",
                            "IS_C_INVERTED": "1",
                            "SLICE": "SLICE_X0Y0"
                        },
                        "attributes": {
                            "src": "design.v:10"
                        },
                        "port_directions": {},
                        "connections": {}
                    }
                },
                "netnames": {}
            }
        }
    }"#;
    let pool = make_pool();
    let design = parse_json(json, &pool).unwrap();

    let cell_id = pool.intern("dff0");
    let cell_idx = design.cell_by_name(cell_id).unwrap();
    let cell = design.cell(cell_idx);

    assert_eq!(cell.params.len(), 3);

    let init = cell.params.get(&pool.intern("INIT")).unwrap();
    assert_eq!(init, &Property::bit_vector("0"));

    let inv = cell.params.get(&pool.intern("IS_C_INVERTED")).unwrap();
    assert_eq!(inv, &Property::bit_vector("1"));

    // "SLICE_X0Y0" contains non-binary characters, so it's a string
    let slice = cell.params.get(&pool.intern("SLICE")).unwrap();
    assert_eq!(slice, &Property::string("SLICE_X0Y0"));
}

// =========================================================================
// Net naming test
// =========================================================================

#[test]
fn test_netnames_rename_nets() {
    let pool = make_pool();
    let design = parse_json(BLINKY_JSON, &pool).unwrap();

    // After parsing, nets originally named $signal$2 and $signal$3
    // should be renamed to "clk" and "led" respectively
    let clk_id = pool.intern("clk");
    let led_id = pool.intern("led");

    let clk_net_idx = design.net_by_name(clk_id);
    assert!(
        clk_net_idx.is_some(),
        "Net 'clk' should exist after renaming"
    );

    let led_net_idx = design.net_by_name(led_id);
    assert!(
        led_net_idx.is_some(),
        "Net 'led' should exist after renaming"
    );

    // The actual NetInfo objects should have the renamed names
    let clk_net = design.net(clk_net_idx.unwrap());
    assert_eq!(clk_net.name, clk_id);

    let led_net = design.net(led_net_idx.unwrap());
    assert_eq!(led_net.name, led_id);
}

// =========================================================================
// Hidden net name test
// =========================================================================

#[test]
fn test_hidden_netnames_not_applied() {
    let json = r#"{
        "modules": {
            "top": {
                "attributes": { "top": "00000000000000000000000000000001" },
                "parameter_default_values": {},
                "ports": {
                    "a": { "direction": "input", "bits": [2] }
                },
                "cells": {},
                "netnames": {
                    "$internal_wire": {
                        "hide_name": 1,
                        "bits": [2],
                        "attributes": {}
                    }
                }
            }
        }
    }"#;
    let pool = make_pool();
    let design = parse_json(json, &pool).unwrap();

    // The hidden name should not be applied
    let hidden_id = pool.intern("$internal_wire");
    assert!(
        design.net_by_name(hidden_id).is_none(),
        "Hidden net name should not be in lookup"
    );
}

// =========================================================================
// Complex design test (two cells, shared nets)
// =========================================================================

#[test]
fn test_two_cells_shared_net() {
    let json = r#"{
        "modules": {
            "top": {
                "attributes": { "top": "00000000000000000000000000000001" },
                "parameter_default_values": {},
                "ports": {},
                "cells": {
                    "src_cell": {
                        "hide_name": 0,
                        "type": "SRC",
                        "parameters": {},
                        "attributes": {},
                        "port_directions": { "O": "output" },
                        "connections": { "O": [42] }
                    },
                    "sink_cell": {
                        "hide_name": 0,
                        "type": "SINK",
                        "parameters": {},
                        "attributes": {},
                        "port_directions": { "I": "input" },
                        "connections": { "I": [42] }
                    }
                },
                "netnames": {
                    "wire_42": {
                        "hide_name": 0,
                        "bits": [42],
                        "attributes": {}
                    }
                }
            }
        }
    }"#;
    let pool = make_pool();
    let design = parse_json(json, &pool).unwrap();

    // Both cells share net 42 (renamed to "wire_42")
    let wire_name = pool.intern("wire_42");
    let net_idx = design.net_by_name(wire_name).unwrap();
    let net = design.net(net_idx);

    // src_cell drives it
    let src_id = pool.intern("src_cell");
    let src_idx = design.cell_by_name(src_id).unwrap();
    assert_eq!(net.driver.cell, Some(src_idx));

    // sink_cell uses it
    let sink_id = pool.intern("sink_cell");
    let sink_idx = design.cell_by_name(sink_id).unwrap();
    assert!(net.users.iter().any(|u| u.cell == Some(sink_idx)));
}

// =========================================================================
// Netnames attributes test
// =========================================================================

#[test]
fn test_netnames_attributes() {
    let json = r#"{
        "modules": {
            "top": {
                "attributes": { "top": "00000000000000000000000000000001" },
                "parameter_default_values": {},
                "ports": {
                    "clk": { "direction": "input", "bits": [2] }
                },
                "cells": {},
                "netnames": {
                    "clk": {
                        "hide_name": 0,
                        "bits": [2],
                        "attributes": {
                            "src": "design.v:1"
                        }
                    }
                }
            }
        }
    }"#;
    let pool = make_pool();
    let design = parse_json(json, &pool).unwrap();

    let clk_id = pool.intern("clk");
    let net_idx = design.net_by_name(clk_id).unwrap();
    let net = design.net(net_idx);

    let src_key = pool.intern("src");
    let attr = net.attrs.get(&src_key).unwrap();
    assert_eq!(attr.as_str(), "design.v:1");
}

// =========================================================================
// Property edge cases
// =========================================================================

#[test]
fn test_property_empty_binary_string() {
    // Empty string that looks binary (all 0s and 1s) but is empty
    let val = serde_json::json!("");
    let prop = parse_property(&val).unwrap();
    // Empty string: all chars vacuously satisfy being 0/1, but is_empty check
    // makes it a string
    assert!(prop.is_string());
}

#[test]
fn test_parse_property_unsupported_type() {
    let val = serde_json::json!(true);
    assert!(parse_property(&val).is_err());
}
