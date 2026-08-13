//! Helper types and functions for Yosys JSON netlist parsing.

use crate::common::IdString;
use crate::netlist::{CellId, Design, NetId, PortType, Property};
use anyhow::{bail, Context, Result};
use rustc_hash::{FxHashMap, FxHashSet};
use serde_json::Value;

/// A bit value in a Yosys JSON connection or port bits array.
///
/// Yosys represents connection bits as either integers (net indices) or
/// string constants ("0", "1", "x"/"z").
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BitValue {
    /// A signal net identified by a unique integer index.
    Signal(i64),
    /// Constant logic 0.
    Zero,
    /// Constant logic 1.
    One,
    /// Don't care / unconnected / high-impedance.
    Undef,
}

/// Parse a single element from a Yosys `bits` or `connections` array.
pub fn parse_bit_value(val: &Value) -> Result<BitValue> {
    match val {
        Value::Number(n) => {
            let idx = n.as_i64().context("Bit index is not a valid integer")?;
            Ok(BitValue::Signal(idx))
        }
        Value::String(s) => match s.as_str() {
            "0" => Ok(BitValue::Zero),
            "1" => Ok(BitValue::One),
            "x" | "z" => Ok(BitValue::Undef),
            other => bail!("Unknown constant bit value: {:?}", other),
        },
        _ => bail!("Invalid bit value in connections array: {:?}", val),
    }
}

/// Infer port direction from cell type and port name when `port_directions`
/// is missing from the Yosys JSON (common with BLIF import).
pub fn infer_port_direction(cell_type: &str, port_name: &str) -> PortType {
    match cell_type {
        "LUT4" | "LUT6" => {
            if port_name == "F" {
                PortType::Out
            } else {
                PortType::In
            }
        }
        "CARRY4" => {
            if port_name.starts_with("CO") || port_name.starts_with("O") {
                PortType::Out
            } else {
                PortType::In
            }
        }
        "GND" | "VCC" | "GND_DRV" | "VCC_DRV" => PortType::Out,
        "IOB" => PortType::InOut,
        _ => {
            // Heuristic: common output port names across LUT, DFF, BUF, IBUF, OBUF, etc.
            if port_name == "Q" || port_name == "O" || port_name == "F" || port_name == "Y" {
                PortType::Out
            } else {
                PortType::In
            }
        }
    }
}

/// Parse a Yosys port direction string into a [`PortType`].
pub fn parse_port_direction(dir: &str) -> Result<PortType> {
    match dir {
        "input" => Ok(PortType::In),
        "output" => Ok(PortType::Out),
        "inout" => Ok(PortType::InOut),
        other => bail!("Unknown port direction: {:?}", other),
    }
}

/// Parse a Yosys property value (parameter or attribute) into a [`Property`].
///
/// Yosys represents parameters as:
///  - Binary strings like `"0000000000001111"` for LUT INIT values
///  - Decimal integer strings
///  - Arbitrary strings (attributes like `"src": "blinky.v:5"`)
///  - Integer JSON values
///
/// We try to determine if a string is a binary bit-vector (only '0'/'1' chars
/// and either long or explicitly bit-patterned) or a plain string/number.
pub fn parse_property(val: &Value) -> Result<Property> {
    match val {
        Value::Number(n) => {
            let v = n.as_i64().context("Property number is not a valid i64")?;
            Ok(Property::int(v))
        }
        Value::String(s) => {
            // Yosys encodes parameters as binary strings of '0' and '1'.
            // Attributes can be binary-encoded integers (32+ chars of 0/1)
            // or plain strings.
            if !s.is_empty() && s.chars().all(|c| c == '0' || c == '1') {
                Ok(Property::bit_vector(s.clone()))
            } else {
                Ok(Property::string(s.clone()))
            }
        }
        _ => bail!("Unsupported property value type: {:?}", val),
    }
}

/// Compute the port name for a potentially multi-bit port.
///
/// If a port has a single bit, its name is used as-is.
/// If a port has multiple bits, each bit gets an indexed name: `port[0]`, `port[1]`, etc.
pub fn port_bit_name(base_name: &str, bit_index: usize, total_bits: usize) -> String {
    if total_bits == 1 {
        base_name.to_string()
    } else {
        format!("{}[{}]", base_name, bit_index)
    }
}

/// Scan the module JSON to collect all unique signal bit indices and create a
/// net for each one. Returns a mapping from bit index to `NetIdx`.
pub fn collect_bit_indices(
    module: &Value,
    bit_to_net: &mut FxHashMap<i64, NetId>,
    design: &mut Design,
    pool: &crate::common::IdStringPool,
) -> Result<()> {
    // Collect from cell connections
    if let Some(cells) = module.get("cells").and_then(|c| c.as_object()) {
        for (_cell_name, cell_json) in cells {
            if let Some(conns) = cell_json.get("connections").and_then(|c| c.as_object()) {
                for (_port_name, bits_val) in conns {
                    let bits = bits_val
                        .as_array()
                        .context("Cell connection bits must be an array")?;
                    for bit in bits {
                        if let BitValue::Signal(idx) = parse_bit_value(bit)? {
                            if !bit_to_net.contains_key(&idx) {
                                let net_name = format!("$signal${}", idx);
                                let net_idx = design.add_net(pool.intern(&net_name));
                                bit_to_net.insert(idx, net_idx);
                            }
                        }
                    }
                }
            }
        }
    }

    // Collect from top-level port bits
    if let Some(ports) = module.get("ports").and_then(|p| p.as_object()) {
        for (_port_name, port_json) in ports {
            if let Some(bits) = port_json.get("bits").and_then(|b| b.as_array()) {
                for bit in bits {
                    if let BitValue::Signal(idx) = parse_bit_value(bit)? {
                        if !bit_to_net.contains_key(&idx) {
                            let net_name = format!("$signal${}", idx);
                            let net_idx = design.add_net(pool.intern(&net_name));
                            bit_to_net.insert(idx, net_idx);
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

/// Create a constant net (GND or VCC).
pub fn create_constant_net(
    design: &mut Design,
    pool: &crate::common::IdStringPool,
    name: &str,
) -> NetId {
    design.add_net(pool.intern(name))
}

/// Create a constant driver cell (GND or VCC) and connect its output to the
/// given net.
pub fn create_constant_driver(
    design: &mut Design,
    pool: &crate::common::IdStringPool,
    cell_name: &str,
    cell_type: &str,
    output_port: &str,
    net_idx: NetId,
) -> Result<()> {
    let name_id = pool.intern(cell_name);
    let type_id = pool.intern(cell_type);
    let port_id = pool.intern(output_port);

    let cell_idx = design.add_cell(name_id, type_id);

    // Add the output port
    design.cell_edit(cell_idx).add_port(port_id, PortType::Out);

    // Connect: set the port's net and set the net's driver
    design
        .cell_edit(cell_idx)
        .set_port_net(port_id, Some(net_idx), None);
    design.net_edit(net_idx).set_driver(cell_idx, port_id);

    Ok(())
}

/// Connect a port on a cell to a net, updating both the port and the net
/// (driver or user reference).
pub fn connect_port_to_net(
    design: &mut Design,
    cell_idx: CellId,
    port_id: IdString,
    port_type: PortType,
    net_idx: NetId,
) -> Result<()> {
    match port_type {
        PortType::Out => {
            // Output port: this port drives the net.
            design
                .cell_edit(cell_idx)
                .set_port_net(port_id, Some(net_idx), None);
            design.net_edit(net_idx).set_driver(cell_idx, port_id);
        }
        PortType::In | PortType::InOut => {
            // Input or bidirectional port: this port is a user of the net.
            let user_idx = design.net_edit(net_idx).add_user(cell_idx, port_id);
            design
                .cell_edit(cell_idx)
                .set_port_net(port_id, Some(net_idx), Some(user_idx));
        }
    }

    Ok(())
}

/// Is `a` the better primary name for a net than `b`?
///
/// Faithful port of `prefer_netlabel` in nextpnr `frontend/frontend_base.h`
/// (upstream YosysHQ `main` @ `4d235150`):
///
/// - top-level ports always win
/// - then fewer `$`
/// - then fewer `.`
/// - then alphabetical
///
/// `top_ports` holds the *bus* names from the JSON `ports` section, matching
/// nextpnr's `port_to_bus` lookup: a per-bit label like `din[5]` therefore only
/// hits this check on single-bit ports. That quirk is reproduced deliberately.
fn prefer_netlabel(a: &str, b: &str, top_ports: &FxHashSet<String>) -> bool {
    if top_ports.contains(a) {
        return true;
    }
    if top_ports.contains(b) {
        return false;
    }
    if b.is_empty() {
        return true;
    }

    let (a_dollars, b_dollars) = (a.matches('$').count(), b.matches('$').count());
    if a_dollars != b_dollars {
        return a_dollars < b_dollars;
    }
    let (a_dots, b_dots) = (a.matches('.').count(), b.matches('.').count());
    if a_dots != b_dots {
        return a_dots < b_dots;
    }
    a < b
}

/// Apply human-readable names from the `netnames` section to nets.
///
/// Every label is a candidate, including ones Yosys marks `hide_name` -- that
/// flag is emitted by nextpnr's JSON writer but never read back by its frontend,
/// so honouring it here would drop the majority of net names (all the
/// `$abc$...`/`$auto$...` ones) and replace them with synthetic `$signal$N`.
/// Where several labels alias the same net, the primary name is chosen by
/// [`prefer_netlabel`] rather than by whichever happened to be visited last.
pub fn apply_net_names(
    netnames: &serde_json::Map<String, Value>,
    design: &mut Design,
    pool: &crate::common::IdStringPool,
    bit_to_net: &FxHashMap<i64, NetId>,
    top_ports: &FxHashSet<String>,
) -> Result<()> {
    let mut candidates: FxHashMap<NetId, Vec<String>> = FxHashMap::default();

    for (net_name, nn_json) in netnames {
        let bits = nn_json
            .get("bits")
            .and_then(|b| b.as_array())
            .context("netnames entry missing 'bits'")?;

        let total_bits = bits.len();

        for (i, bit) in bits.iter().enumerate() {
            if let BitValue::Signal(idx) = parse_bit_value(bit)? {
                if let Some(&net_idx) = bit_to_net.get(&idx) {
                    candidates
                        .entry(net_idx)
                        .or_default()
                        .push(port_bit_name(net_name, i, total_bits));
                }
            }
        }

        // Apply attributes to the net
        if let Some(attrs) = nn_json.get("attributes").and_then(|a| a.as_object()) {
            // Apply to the first signal bit's net (if it exists)
            if let Some(first_bit) = bits.first() {
                if let BitValue::Signal(idx) = parse_bit_value(first_bit)? {
                    if let Some(&net_idx) = bit_to_net.get(&idx) {
                        for (attr_name, attr_val) in attrs {
                            let key = pool.intern(attr_name);
                            let prop = parse_property(attr_val)?;
                            design.net_edit(net_idx).set_attr(key, prop);
                        }
                    }
                }
            }
        }
    }

    // Pick each net's primary name once every candidate label is known.
    for (net_idx, names) in candidates {
        let mut best = names[0].as_str();
        for name in &names[1..] {
            if prefer_netlabel(name, best, top_ports) {
                best = name;
            }
        }
        let name_id = pool.intern(best);
        design.rename_net(net_idx, name_id);
    }

    Ok(())
}
