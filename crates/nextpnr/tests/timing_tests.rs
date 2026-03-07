use nextpnr::netlist::{CellId, Design, NetId};
use nextpnr::timing::{topological_sort, ClockDomain, TimingAnalyser};
use nextpnr::types::{ClockEdge, IdString, IdStringPool, PortType, TimingPortClass};
use std::collections::HashSet;

// =========================================================================
// Test helpers from lib.rs
// =========================================================================

/// Helper to create a design with a simple combinational chain:
/// INPUT -> LUT_A -> LUT_B -> OUTPUT
fn make_comb_chain_design(pool: &IdStringPool) -> Design {
    let mut d = Design::new();

    // Cell types
    let io_type = pool.intern("IO");
    let lut_type = pool.intern("LUT4");

    // Cell names
    let input_name = pool.intern("input_cell");
    let lut_a_name = pool.intern("lut_a");
    let lut_b_name = pool.intern("lut_b");
    let output_name = pool.intern("output_cell");

    // Port names
    let i_port = pool.intern("I");
    let o_port = pool.intern("O");
    let a_port = pool.intern("A");
    let f_port = pool.intern("F");

    // Net names
    let net_in = pool.intern("net_in");
    let net_ab = pool.intern("net_ab");
    let net_out = pool.intern("net_out");

    // Create cells
    let input_idx = d.add_cell(input_name, io_type);
    let lut_a_idx = d.add_cell(lut_a_name, lut_type);
    let lut_b_idx = d.add_cell(lut_b_name, lut_type);
    let output_idx = d.add_cell(output_name, io_type);

    // Add ports
    d.cell_edit(input_idx).add_port(o_port, PortType::Out);
    d.cell_edit(lut_a_idx).add_port(a_port, PortType::In);
    d.cell_edit(lut_a_idx).add_port(f_port, PortType::Out);
    d.cell_edit(lut_b_idx).add_port(a_port, PortType::In);
    d.cell_edit(lut_b_idx).add_port(f_port, PortType::Out);
    d.cell_edit(output_idx).add_port(i_port, PortType::In);

    // Create nets and wire them up.
    // net_in: input_cell.O -> lut_a.A
    let net_in_idx = d.add_net(net_in);
    d.net_edit(net_in_idx).set_driver(input_idx, o_port);
    d.cell_edit(input_idx)
        .set_port_net(o_port, Some(net_in_idx), None);

    let user_idx = d.net_edit(net_in_idx).add_user(lut_a_idx, a_port);
    d.cell_edit(lut_a_idx)
        .set_port_net(a_port, Some(net_in_idx), Some(user_idx));

    // net_ab: lut_a.F -> lut_b.A
    let net_ab_idx = d.add_net(net_ab);
    d.net_edit(net_ab_idx).set_driver(lut_a_idx, f_port);
    d.cell_edit(lut_a_idx)
        .set_port_net(f_port, Some(net_ab_idx), None);

    let user_idx = d.net_edit(net_ab_idx).add_user(lut_b_idx, a_port);
    d.cell_edit(lut_b_idx)
        .set_port_net(a_port, Some(net_ab_idx), Some(user_idx));

    // net_out: lut_b.F -> output_cell.I
    let net_out_idx = d.add_net(net_out);
    d.net_edit(net_out_idx).set_driver(lut_b_idx, f_port);
    d.cell_edit(lut_b_idx)
        .set_port_net(f_port, Some(net_out_idx), None);

    let user_idx = d.net_edit(net_out_idx).add_user(output_idx, i_port);
    d.cell_edit(output_idx)
        .set_port_net(i_port, Some(net_out_idx), Some(user_idx));

    d
}

/// Helper to create a register-to-register design:
/// FF_A.Q -> LUT -> FF_B.D
/// with a clock net "clk" connected to FF_A.CLK and FF_B.CLK.
fn make_reg_to_reg_design(pool: &IdStringPool) -> (Design, IdString) {
    let mut d = Design::new();

    let ff_type = pool.intern("FF");
    let lut_type = pool.intern("LUT4");

    let ff_a_name = pool.intern("ff_a");
    let lut_name = pool.intern("lut_mid");
    let ff_b_name = pool.intern("ff_b");

    let clk_port = pool.intern("CLK");
    let d_port = pool.intern("D");
    let q_port = pool.intern("Q");
    let a_port = pool.intern("A");
    let f_port = pool.intern("F");

    let clk_net_name = pool.intern("clk");
    let net_q = pool.intern("net_q");
    let net_f = pool.intern("net_f");

    // Create cells
    let ff_a_idx = d.add_cell(ff_a_name, ff_type);
    let lut_idx = d.add_cell(lut_name, lut_type);
    let ff_b_idx = d.add_cell(ff_b_name, ff_type);

    // Add ports
    d.cell_edit(ff_a_idx).add_port(clk_port, PortType::In);
    d.cell_edit(ff_a_idx).add_port(d_port, PortType::In);
    d.cell_edit(ff_a_idx).add_port(q_port, PortType::Out);

    d.cell_edit(lut_idx).add_port(a_port, PortType::In);
    d.cell_edit(lut_idx).add_port(f_port, PortType::Out);

    d.cell_edit(ff_b_idx).add_port(clk_port, PortType::In);
    d.cell_edit(ff_b_idx).add_port(d_port, PortType::In);
    d.cell_edit(ff_b_idx).add_port(q_port, PortType::Out);

    // Create clock net with constraint.
    let clk_net_idx = d.add_net(clk_net_name);
    d.net_edit(clk_net_idx).set_clock_constraint(10_000); // 10ns = 100 MHz

    // Connect clock to FF_A.CLK and FF_B.CLK (no driver cell for clk).
    let user_idx = d.net_edit(clk_net_idx).add_user(ff_a_idx, clk_port);
    d.cell_edit(ff_a_idx)
        .set_port_net(clk_port, Some(clk_net_idx), Some(user_idx));

    let user_idx = d.net_edit(clk_net_idx).add_user(ff_b_idx, clk_port);
    d.cell_edit(ff_b_idx)
        .set_port_net(clk_port, Some(clk_net_idx), Some(user_idx));

    // net_q: FF_A.Q -> LUT.A
    let net_q_idx = d.add_net(net_q);
    d.net_edit(net_q_idx).set_driver(ff_a_idx, q_port);
    d.cell_edit(ff_a_idx)
        .set_port_net(q_port, Some(net_q_idx), None);

    let user_idx = d.net_edit(net_q_idx).add_user(lut_idx, a_port);
    d.cell_edit(lut_idx)
        .set_port_net(a_port, Some(net_q_idx), Some(user_idx));

    // net_f: LUT.F -> FF_B.D
    let net_f_idx = d.add_net(net_f);
    d.net_edit(net_f_idx).set_driver(lut_idx, f_port);
    d.cell_edit(lut_idx)
        .set_port_net(f_port, Some(net_f_idx), None);

    let user_idx = d.net_edit(net_f_idx).add_user(ff_b_idx, d_port);
    d.cell_edit(ff_b_idx)
        .set_port_net(d_port, Some(net_f_idx), Some(user_idx));

    (d, clk_net_name)
}

/// Helper to create a two-clock-domain design:
/// FF_A (clk1) -> LUT -> FF_B (clk2)
fn make_two_domain_design(pool: &IdStringPool) -> Design {
    let mut d = Design::new();

    let ff_type = pool.intern("FF");
    let lut_type = pool.intern("LUT4");

    let ff_a_name = pool.intern("ff_a");
    let lut_name = pool.intern("lut_mid");
    let ff_b_name = pool.intern("ff_b");

    let clk_port = pool.intern("CLK");
    let d_port = pool.intern("D");
    let q_port = pool.intern("Q");
    let a_port = pool.intern("A");
    let f_port = pool.intern("F");

    let clk1_net_name = pool.intern("clk1");
    let clk2_net_name = pool.intern("clk2");
    let net_q = pool.intern("net_q");
    let net_f = pool.intern("net_f");

    // Create cells
    let ff_a_idx = d.add_cell(ff_a_name, ff_type);
    let lut_idx = d.add_cell(lut_name, lut_type);
    let ff_b_idx = d.add_cell(ff_b_name, ff_type);

    // Add ports
    d.cell_edit(ff_a_idx).add_port(clk_port, PortType::In);
    d.cell_edit(ff_a_idx).add_port(d_port, PortType::In);
    d.cell_edit(ff_a_idx).add_port(q_port, PortType::Out);

    d.cell_edit(lut_idx).add_port(a_port, PortType::In);
    d.cell_edit(lut_idx).add_port(f_port, PortType::Out);

    d.cell_edit(ff_b_idx).add_port(clk_port, PortType::In);
    d.cell_edit(ff_b_idx).add_port(d_port, PortType::In);
    d.cell_edit(ff_b_idx).add_port(q_port, PortType::Out);

    // Clock 1 net
    let clk1_idx = d.add_net(clk1_net_name);
    d.net_edit(clk1_idx).set_clock_constraint(10_000); // 100 MHz

    let user_idx = d.net_edit(clk1_idx).add_user(ff_a_idx, clk_port);
    d.cell_edit(ff_a_idx)
        .set_port_net(clk_port, Some(clk1_idx), Some(user_idx));

    // Clock 2 net
    let clk2_idx = d.add_net(clk2_net_name);
    d.net_edit(clk2_idx).set_clock_constraint(5_000); // 200 MHz

    let user_idx = d.net_edit(clk2_idx).add_user(ff_b_idx, clk_port);
    d.cell_edit(ff_b_idx)
        .set_port_net(clk_port, Some(clk2_idx), Some(user_idx));

    // net_q: FF_A.Q -> LUT.A
    let net_q_idx = d.add_net(net_q);
    d.net_edit(net_q_idx).set_driver(ff_a_idx, q_port);
    d.cell_edit(ff_a_idx)
        .set_port_net(q_port, Some(net_q_idx), None);

    let user_idx = d.net_edit(net_q_idx).add_user(lut_idx, a_port);
    d.cell_edit(lut_idx)
        .set_port_net(a_port, Some(net_q_idx), Some(user_idx));

    // net_f: LUT.F -> FF_B.D
    let net_f_idx = d.add_net(net_f);
    d.net_edit(net_f_idx).set_driver(lut_idx, f_port);
    d.cell_edit(lut_idx)
        .set_port_net(f_port, Some(net_f_idx), None);

    let user_idx = d.net_edit(net_f_idx).add_user(ff_b_idx, d_port);
    d.cell_edit(ff_b_idx)
        .set_port_net(d_port, Some(net_f_idx), Some(user_idx));

    d
}

// =========================================================================
// Tests from domain.rs
// =========================================================================

#[test]
fn unclocked_domain() {
    let d = ClockDomain::unclocked();
    assert!(!d.is_clocked());
    assert!(d.clock_net.is_empty());
    assert_eq!(d.period, 0);
    assert_eq!(d.edge, ClockEdge::Rising);
}

#[test]
fn default_is_unclocked() {
    let d = ClockDomain::default();
    assert!(!d.is_clocked());
}

#[test]
fn clocked_domain() {
    let pool = IdStringPool::new();
    let clk = pool.intern("sys_clk");
    let d = ClockDomain {
        clock_net: clk,
        edge: ClockEdge::Falling,
        period: 5000,
    };
    assert!(d.is_clocked());
    assert_eq!(d.period, 5000);
    assert_eq!(d.edge, ClockEdge::Falling);
}

#[test]
fn domain_equality() {
    let pool = IdStringPool::new();
    let clk = pool.intern("clk");
    let a = ClockDomain {
        clock_net: clk,
        edge: ClockEdge::Rising,
        period: 10_000,
    };
    let b = ClockDomain {
        clock_net: clk,
        edge: ClockEdge::Rising,
        period: 10_000,
    };
    let c = ClockDomain {
        clock_net: clk,
        edge: ClockEdge::Falling,
        period: 10_000,
    };
    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn domain_hashing() {
    let pool = IdStringPool::new();
    let clk = pool.intern("clk");
    let d1 = ClockDomain {
        clock_net: clk,
        edge: ClockEdge::Rising,
        period: 10_000,
    };
    let d2 = ClockDomain {
        clock_net: clk,
        edge: ClockEdge::Rising,
        period: 10_000,
    };
    let d3 = ClockDomain::unclocked();

    let mut set = HashSet::new();
    set.insert(d1);
    set.insert(d2);
    set.insert(d3);
    assert_eq!(set.len(), 2);
}

#[test]
fn domain_clone() {
    let pool = IdStringPool::new();
    let clk = pool.intern("fast_clk");
    let d = ClockDomain {
        clock_net: clk,
        edge: ClockEdge::Rising,
        period: 2000,
    };
    let d2 = d.clone();
    assert_eq!(d, d2);
}

// =========================================================================
// Tests from sort.rs
// =========================================================================

#[test]
fn sort_empty_design() {
    let design = Design::new();
    let sorted = topological_sort(&design);
    assert!(sorted.is_empty());
}

#[test]
fn sort_single_cell() {
    let pool = IdStringPool::new();
    let mut design = Design::new();
    let name = pool.intern("cell_a");
    let ctype = pool.intern("LUT4");
    let idx = design.add_cell(name, ctype);

    let sorted = topological_sort(&design);
    assert_eq!(sorted.len(), 1);
    assert_eq!(sorted[0], idx);
}

#[test]
fn sort_chain_of_three() {
    let pool = IdStringPool::new();
    let mut d = Design::new();

    let a_name = pool.intern("a");
    let b_name = pool.intern("b");
    let c_name = pool.intern("c");
    let lut = pool.intern("LUT4");

    let o = pool.intern("O");
    let i = pool.intern("I");

    let a_idx = d.add_cell(a_name, lut);
    let b_idx = d.add_cell(b_name, lut);
    let c_idx = d.add_cell(c_name, lut);

    d.cell_edit(a_idx).add_port(o, PortType::Out);
    d.cell_edit(b_idx).add_port(i, PortType::In);
    d.cell_edit(b_idx).add_port(o, PortType::Out);
    d.cell_edit(c_idx).add_port(i, PortType::In);

    // a.O -> b.I
    let n1_name = pool.intern("n1");
    let n1 = d.add_net(n1_name);
    d.net_edit(n1).set_driver(a_idx, o);
    d.cell_edit(a_idx).set_port_net(o, Some(n1), None);
    d.net_edit(n1).add_user(b_idx, i);
    d.cell_edit(b_idx).set_port_net(i, Some(n1), None);

    // b.O -> c.I
    let n2_name = pool.intern("n2");
    let n2 = d.add_net(n2_name);
    d.net_edit(n2).set_driver(b_idx, o);
    d.cell_edit(b_idx).set_port_net(o, Some(n2), None);
    d.net_edit(n2).add_user(c_idx, i);
    d.cell_edit(c_idx).set_port_net(i, Some(n2), None);

    let sorted = topological_sort(&d);
    assert_eq!(sorted.len(), 3);

    let a_pos = sorted.iter().position(|&x| x == a_idx).unwrap();
    let b_pos = sorted.iter().position(|&x| x == b_idx).unwrap();
    let c_pos = sorted.iter().position(|&x| x == c_idx).unwrap();

    assert!(a_pos < b_pos, "a should come before b");
    assert!(b_pos < c_pos, "b should come before c");
}

#[test]
fn sort_fanout() {
    let pool = IdStringPool::new();
    let mut d = Design::new();

    let lut = pool.intern("LUT4");
    let o = pool.intern("O");
    let i = pool.intern("I");

    let a_idx = d.add_cell(pool.intern("a"), lut);
    let b_idx = d.add_cell(pool.intern("b"), lut);
    let c_idx = d.add_cell(pool.intern("c"), lut);

    d.cell_edit(a_idx).add_port(o, PortType::Out);
    d.cell_edit(b_idx).add_port(i, PortType::In);
    d.cell_edit(c_idx).add_port(i, PortType::In);

    // a.O -> b.I and a.O -> c.I (fanout of 2)
    let n1 = d.add_net(pool.intern("n1"));
    d.net_edit(n1).set_driver(a_idx, o);
    d.cell_edit(a_idx).set_port_net(o, Some(n1), None);
    d.net_edit(n1).add_user(b_idx, i);
    d.cell_edit(b_idx).set_port_net(i, Some(n1), None);
    d.net_edit(n1).add_user(c_idx, i);
    d.cell_edit(c_idx).set_port_net(i, Some(n1), None);

    let sorted = topological_sort(&d);
    assert_eq!(sorted.len(), 3);

    let a_pos = sorted.iter().position(|&x| x == a_idx).unwrap();
    let b_pos = sorted.iter().position(|&x| x == b_idx).unwrap();
    let c_pos = sorted.iter().position(|&x| x == c_idx).unwrap();

    assert!(a_pos < b_pos, "a should come before b");
    assert!(a_pos < c_pos, "a should come before c");
}

#[test]
fn sort_dead_cells_excluded() {
    let pool = IdStringPool::new();
    let mut d = Design::new();

    let lut = pool.intern("LUT4");
    let a_name = pool.intern("a");
    let b_name = pool.intern("b");

    d.add_cell(a_name, lut);
    d.add_cell(b_name, lut);

    // Kill cell b.
    d.remove_cell(b_name);

    let sorted = topological_sort(&d);
    assert_eq!(sorted.len(), 1);
    assert_eq!(sorted[0], CellId::from_raw(0));
}

// =========================================================================
// Tests from lib.rs (TimingAnalyser)
// =========================================================================

#[test]
fn test_analyser_new() {
    let ta = TimingAnalyser::new();
    assert!(!ta.is_valid());
    assert_eq!(ta.worst_slack(), 0);
    assert_eq!(ta.net_criticality(NetId::from_raw(0)), 0.0);
}

#[test]
fn test_clock_constraint_mhz() {
    let pool = IdStringPool::new();
    let mut ta = TimingAnalyser::new();
    let clk = pool.intern("clk");
    ta.add_clock_constraint(clk, 100.0);
    assert_eq!(*ta.clock_constraints().get(&clk).unwrap(), 10_000); // 10ns = 10000ps
}

#[test]
fn test_clock_constraint_ps() {
    let pool = IdStringPool::new();
    let mut ta = TimingAnalyser::new();
    let clk = pool.intern("clk");
    ta.add_clock_constraint_ps(clk, 5000);
    assert_eq!(*ta.clock_constraints().get(&clk).unwrap(), 5000);
}

#[test]
fn test_comb_chain_forward_propagation() {
    let pool = IdStringPool::new();
    let design = make_comb_chain_design(&pool);

    let mut ta = TimingAnalyser::new();
    ta.analyse(&design, &pool);

    assert!(ta.is_valid());

    // The combinational chain: input -> lut_a -> lut_b -> output
    // All ports are combinational (no clock).
    // input.O: arrival = 0 (primary input)
    // lut_a.A: arrival = 0 (from input.O, net delay 0)
    // lut_a.F: arrival = 0 + 100 = 100 (comb delay)
    // lut_b.A: arrival = 100 (from lut_a.F)
    // lut_b.F: arrival = 100 + 100 = 200
    // output.I: arrival = 200

    let o_port = pool.intern("O");
    let f_port = pool.intern("F");

    let input_idx = CellId::from_raw(0);
    let lut_a_idx = CellId::from_raw(1);
    let lut_b_idx = CellId::from_raw(2);

    // Check arrival times.
    // input.O is a pure source cell with no inputs: arrival = 0.
    assert_eq!(
        ta.arrival_time(input_idx, o_port),
        Some(0),
        "Input output port should have arrival 0 (primary source)"
    );

    // lut_a.F should have arrival = 0 (from input.O) + 100 (comb delay) = 100.
    let lut_a_f_arrival = ta.arrival_time(lut_a_idx, f_port);
    assert!(
        lut_a_f_arrival.is_some(),
        "lut_a.F should have an arrival time"
    );
    assert_eq!(lut_a_f_arrival.unwrap(), 100);

    // lut_b.F should have arrival = 100 + 100 = 200.
    let lut_b_f_arrival = ta.arrival_time(lut_b_idx, f_port);
    assert!(
        lut_b_f_arrival.is_some(),
        "lut_b.F should have an arrival time"
    );
    assert_eq!(lut_b_f_arrival.unwrap(), 200);
}

#[test]
fn test_reg_to_reg_timing() {
    let pool = IdStringPool::new();
    let (design, _clk_net_name) = make_reg_to_reg_design(&pool);

    let mut ta = TimingAnalyser::new();
    ta.analyse(&design, &pool);

    assert!(ta.is_valid());

    let d_port = pool.intern("D");
    let q_port = pool.intern("Q");
    let a_port = pool.intern("A");
    let f_port = pool.intern("F");

    let ff_a_idx = CellId::from_raw(0);
    let lut_idx = CellId::from_raw(1);
    let ff_b_idx = CellId::from_raw(2);

    // FF_A.Q is a register output: arrival = DEFAULT_COMB_DELAY = 100
    let ff_a_q_arrival = ta.arrival_time(ff_a_idx, q_port);
    assert_eq!(ff_a_q_arrival, Some(100));

    // LUT.A gets arrival from FF_A.Q (100) + net delay (0) = 100
    let lut_a_arrival = ta.arrival_time(lut_idx, a_port);
    assert_eq!(lut_a_arrival, Some(100));

    // LUT.F: arrival = 100 + 100 = 200
    let lut_f_arrival = ta.arrival_time(lut_idx, f_port);
    assert_eq!(lut_f_arrival, Some(200));

    // FF_B.D gets arrival from LUT.F (200) + net delay (0) = 200
    let ff_b_d_arrival = ta.arrival_time(ff_b_idx, d_port);
    assert_eq!(ff_b_d_arrival, Some(200));

    // FF_B.D is a register input.
    // Required time = clock period - setup = 10000 - 50 = 9950
    let ff_b_d_required = ta.required_time(ff_b_idx, d_port);
    assert_eq!(ff_b_d_required, Some(9950));

    // Slack = required - arrival = 9950 - 200 = 9750 (positive = timing met)
    assert!(ta.worst_slack() > 0);

    // Paths should exist.
    assert!(!ta.paths().is_empty());
}

#[test]
fn test_reg_to_reg_tight_constraint() {
    let pool = IdStringPool::new();
    let (mut design, clk_net_name) = make_reg_to_reg_design(&pool);

    // Tighten the clock constraint to 150ps (very tight, will fail).
    let clk_net_idx = design.net_by_name(clk_net_name).unwrap();
    design.net_edit(clk_net_idx).set_clock_constraint(150);

    let mut ta = TimingAnalyser::new();
    ta.analyse(&design, &pool);

    // FF_A.Q arrival = 100
    // LUT.F arrival = 200
    // FF_B.D arrival = 200
    // FF_B.D required = 150 - 50 = 100
    // Slack = 100 - 200 = -100 (negative = timing violated)
    assert!(ta.worst_slack() < 0);

    // With failing timing, criticality should be non-zero.
    let report = ta.report();
    assert!(report.num_failing > 0);
}

#[test]
fn test_two_clock_domains() {
    let pool = IdStringPool::new();
    let design = make_two_domain_design(&pool);

    let mut ta = TimingAnalyser::new();
    ta.analyse(&design, &pool);

    assert!(ta.is_valid());

    let clk_port = pool.intern("CLK");
    let ff_a_idx = CellId::from_raw(0);
    let ff_b_idx = CellId::from_raw(2);

    // Check that FF_A's clock port is classified as ClockInput.
    let ff_a_clk_class = ta.port_class(ff_a_idx, clk_port);
    assert_eq!(ff_a_clk_class, Some(TimingPortClass::ClockInput));

    // Check that FF_B's clock port is classified as ClockInput.
    let ff_b_clk_class = ta.port_class(ff_b_idx, clk_port);
    assert_eq!(ff_b_clk_class, Some(TimingPortClass::ClockInput));

    // Check that the domains are different.
    let ff_a_domain = ta.port_domain(ff_a_idx, clk_port);
    let ff_b_domain = ta.port_domain(ff_b_idx, clk_port);

    assert!(ff_a_domain.is_some());
    assert!(ff_b_domain.is_some());
    assert_ne!(
        ff_a_domain.unwrap().clock_net,
        ff_b_domain.unwrap().clock_net
    );
}

#[test]
fn test_criticality_all_met() {
    let pool = IdStringPool::new();
    let (design, _) = make_reg_to_reg_design(&pool);

    let mut ta = TimingAnalyser::new();
    ta.analyse(&design, &pool);

    // With 10ns period and ~200ps delay, timing is easily met.
    // All criticalities should be 0.
    for (net_idx, _net) in design.iter_nets() {
        let crit = ta.net_criticality(net_idx);
        assert_eq!(
            crit,
            0.0,
            "Net {} should have criticality 0 when timing is met",
            net_idx.raw()
        );
    }
}

#[test]
fn test_criticality_failing() {
    let pool = IdStringPool::new();
    let (mut design, clk_net_name) = make_reg_to_reg_design(&pool);

    // Set very tight constraint to cause timing failure.
    let clk_net_idx = design.net_by_name(clk_net_name).unwrap();
    design.net_edit(clk_net_idx).set_clock_constraint(150);

    let mut ta = TimingAnalyser::new();
    ta.analyse(&design, &pool);

    assert!(ta.worst_slack() < 0, "Should have negative slack");

    // At least one net should have non-zero criticality.
    let mut has_nonzero_crit = false;
    for (net_idx, _net) in design.iter_nets() {
        let crit = ta.net_criticality(net_idx);
        if crit > 0.0 {
            has_nonzero_crit = true;
        }
        assert!(
            crit >= 0.0 && crit <= 1.0,
            "Criticality must be in [0,1], got {}",
            crit
        );
    }
    assert!(
        has_nonzero_crit,
        "Should have at least one net with non-zero criticality"
    );
}

#[test]
fn test_topological_sort_basic() {
    let pool = IdStringPool::new();
    let design = make_comb_chain_design(&pool);

    let sorted = topological_sort(&design);

    // All alive cells should be in the sorted list.
    assert_eq!(sorted.len(), 4);

    // Input cell should come before LUT_A, which should come before LUT_B,
    // which should come before Output.
    let input_pos = sorted
        .iter()
        .position(|&c| c == CellId::from_raw(0))
        .unwrap();
    let lut_a_pos = sorted
        .iter()
        .position(|&c| c == CellId::from_raw(1))
        .unwrap();
    let lut_b_pos = sorted
        .iter()
        .position(|&c| c == CellId::from_raw(2))
        .unwrap();
    let output_pos = sorted
        .iter()
        .position(|&c| c == CellId::from_raw(3))
        .unwrap();

    assert!(
        input_pos < lut_a_pos,
        "input should come before lut_a in topological order"
    );
    assert!(
        lut_a_pos < lut_b_pos,
        "lut_a should come before lut_b in topological order"
    );
    assert!(
        lut_b_pos < output_pos,
        "lut_b should come before output in topological order"
    );
}

#[test]
fn test_topological_sort_with_registers() {
    let pool = IdStringPool::new();
    let (design, _) = make_reg_to_reg_design(&pool);

    let sorted = topological_sort(&design);

    // Should contain all 3 cells.
    assert_eq!(sorted.len(), 3);

    // FF_A should come before LUT (FF_A.Q drives LUT.A).
    // LUT should come before FF_B (LUT.F drives FF_B.D).
    let ff_a_pos = sorted
        .iter()
        .position(|&c| c == CellId::from_raw(0))
        .unwrap();
    let lut_pos = sorted
        .iter()
        .position(|&c| c == CellId::from_raw(1))
        .unwrap();
    let ff_b_pos = sorted
        .iter()
        .position(|&c| c == CellId::from_raw(2))
        .unwrap();

    assert!(
        ff_a_pos < lut_pos,
        "ff_a should come before lut in topological order"
    );
    assert!(
        lut_pos < ff_b_pos,
        "lut should come before ff_b in topological order"
    );
}

#[test]
fn test_invalidate() {
    let pool = IdStringPool::new();
    let design = make_comb_chain_design(&pool);

    let mut ta = TimingAnalyser::new();
    ta.analyse(&design, &pool);
    assert!(ta.is_valid());

    ta.invalidate();
    assert!(!ta.is_valid());
}

#[test]
fn test_fmax_with_constraint() {
    let pool = IdStringPool::new();
    let (design, clk_net_name) = make_reg_to_reg_design(&pool);

    let mut ta = TimingAnalyser::new();
    ta.add_clock_constraint_ps(clk_net_name, 10_000);
    ta.analyse(&design, &pool);

    // Timing should be met with 10ns period.
    let fmax = ta.fmax_mhz();
    assert!(fmax > 0.0, "fmax should be positive");
    // With slack > 0, fmax should equal or exceed the constraint frequency.
    assert!(
        fmax >= 100.0,
        "fmax should be at least 100 MHz with 10ns period"
    );
}

#[test]
fn test_report_structure() {
    let pool = IdStringPool::new();
    let (design, _) = make_reg_to_reg_design(&pool);

    let mut ta = TimingAnalyser::new();
    ta.analyse(&design, &pool);

    let report = ta.report();
    assert!(report.fmax >= 0.0);
    assert_eq!(report.num_failing, 0);
    assert!(report.num_endpoints > 0);
}

#[test]
fn test_empty_design() {
    let pool = IdStringPool::new();
    let design = Design::new();

    let mut ta = TimingAnalyser::new();
    ta.analyse(&design, &pool);

    assert!(ta.is_valid());
    assert_eq!(ta.worst_slack(), 0);
    assert_eq!(ta.fmax_mhz(), 0.0);
}

#[test]
fn test_port_criticality() {
    let pool = IdStringPool::new();
    let (mut design, clk_net_name) = make_reg_to_reg_design(&pool);

    // Use tight constraint.
    let clk_net_idx = design.net_by_name(clk_net_name).unwrap();
    design.net_edit(clk_net_idx).set_clock_constraint(150);

    let mut ta = TimingAnalyser::new();
    ta.analyse(&design, &pool);

    let d_port = pool.intern("D");
    let ff_b_idx = CellId::from_raw(2);

    // FF_B.D is a timing endpoint with failing slack.
    let crit = ta.port_criticality(ff_b_idx, d_port);
    assert!(
        crit > 0.0,
        "FF_B.D should have non-zero criticality with tight constraint"
    );
    assert!(crit <= 1.0, "Criticality should not exceed 1.0");
}

#[test]
fn test_clock_domain_unclocked() {
    let domain = ClockDomain::unclocked();
    assert!(!domain.is_clocked());
    assert!(domain.clock_net.is_empty());
    assert_eq!(domain.period, 0);
}

#[test]
fn test_clock_domain_clocked() {
    let pool = IdStringPool::new();
    let clk = pool.intern("clk");
    let domain = ClockDomain {
        clock_net: clk,
        edge: ClockEdge::Rising,
        period: 10_000,
    };
    assert!(domain.is_clocked());
    assert_eq!(domain.period, 10_000);
}

#[test]
fn test_critical_paths_limit() {
    let pool = IdStringPool::new();
    let (mut design, clk_net_name) = make_reg_to_reg_design(&pool);
    let clk_net_idx = design.net_by_name(clk_net_name).unwrap();
    design.net_edit(clk_net_idx).set_clock_constraint(150);

    let mut ta = TimingAnalyser::new();
    ta.analyse(&design, &pool);

    // Request more paths than exist.
    let paths = ta.critical_paths(100);
    assert!(!paths.is_empty());

    // Request 0 paths.
    let paths = ta.critical_paths(0);
    assert!(paths.is_empty());

    // Request exactly 1 path.
    let paths = ta.critical_paths(1);
    assert_eq!(paths.len(), 1);
}

#[test]
fn test_paths_sorted_by_slack() {
    let pool = IdStringPool::new();
    let (mut design, clk_net_name) = make_reg_to_reg_design(&pool);
    let clk_net_idx = design.net_by_name(clk_net_name).unwrap();
    design.net_edit(clk_net_idx).set_clock_constraint(150);

    let mut ta = TimingAnalyser::new();
    ta.analyse(&design, &pool);

    let paths = ta.critical_paths(100);
    for i in 1..paths.len() {
        assert!(
            paths[i - 1].slack <= paths[i].slack,
            "Paths should be sorted by slack ascending"
        );
    }
}
