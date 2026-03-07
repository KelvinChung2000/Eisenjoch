#![allow(dead_code)]

use nextpnr::chipdb::testutil::make_test_chipdb;
use nextpnr::context::Context;
use nextpnr::types::PortType;

pub fn make_context() -> Context {
    let chipdb = make_test_chipdb();
    Context::new(chipdb)
}

pub fn make_context_with_cells(n: usize) -> Context {
    assert!(n <= 4, "synthetic chipdb only has 4 BELs");
    let mut ctx = make_context();
    ctx.populate_bel_buckets();

    let cell_type = ctx.id("LUT4");
    let mut cell_names = Vec::new();

    for i in 0..n {
        let name = ctx.id(&format!("cell_{}", i));
        ctx.design.add_cell(name, cell_type);
        cell_names.push(name);
    }

    if n >= 2 {
        let net_name = ctx.id("net_0");
        let net_idx = ctx.design.add_net(net_name);
        let q_port = ctx.id("Q");
        let a_port = ctx.id("A");

        let cell0_idx = ctx.design.cell_by_name(cell_names[0]).unwrap();
        ctx.design
            .cell_edit(cell0_idx)
            .add_port(q_port, PortType::Out);
        ctx.design
            .cell_edit(cell0_idx)
            .set_port_net(q_port, Some(net_idx), None);

        ctx.design.net_edit(net_idx).set_driver(cell0_idx, q_port);

        for &name in &cell_names[1..] {
            let cell_idx = ctx.design.cell_by_name(name).unwrap();
            ctx.design
                .cell_edit(cell_idx)
                .add_port(a_port, PortType::In);

            let user_idx = ctx.design.net_edit(net_idx).add_user(cell_idx, a_port);
            ctx.design
                .cell_edit(cell_idx)
                .set_port_net(a_port, Some(net_idx), Some(user_idx));
        }
    }

    ctx
}
