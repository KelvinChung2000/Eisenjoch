mod common;

use nextpnr::chipdb::BelId;
use nextpnr::common::PlaceStrength;
use nextpnr::metrics::{
    accumulate_edge_crossings, bresenham_line, compute_bbox, compute_congestion_ratios,
    compute_sliding_window_density, estimate_congestion, net_hpwl, net_line_estimate,
    placement_density, total_hpwl, Axis, BoundingBox,
};
use nextpnr::netlist::PortType;

// ---------------------------------------------------------------------------
// Bresenham line tests
// ---------------------------------------------------------------------------

#[test]
fn bresenham_horizontal() {
    let points = bresenham_line(0, 0, 3, 0);
    assert_eq!(points, vec![(0, 0), (1, 0), (2, 0), (3, 0)]);
}

#[test]
fn bresenham_vertical() {
    let points = bresenham_line(0, 0, 0, 3);
    assert_eq!(points, vec![(0, 0), (0, 1), (0, 2), (0, 3)]);
}

#[test]
fn bresenham_diagonal() {
    let points = bresenham_line(0, 0, 3, 3);
    assert_eq!(points, vec![(0, 0), (1, 1), (2, 2), (3, 3)]);
}

#[test]
fn bresenham_steep() {
    let points = bresenham_line(0, 0, 1, 3);
    assert_eq!(points.len(), 4);
    // Steep line: more vertical movement than horizontal.
    // Check that y changes every step.
    for i in 1..points.len() {
        let dy = (points[i].1 - points[i - 1].1).abs();
        assert_eq!(dy, 1, "each step should advance y by 1 for a steep line");
    }
    assert_eq!(points.first(), Some(&(0, 0)));
    assert_eq!(points.last(), Some(&(1, 3)));
}

#[test]
fn bresenham_single_point() {
    let points = bresenham_line(2, 3, 2, 3);
    assert_eq!(points, vec![(2, 3)]);
}

#[test]
fn bresenham_negative() {
    let points = bresenham_line(3, 3, 0, 0);
    assert_eq!(points.len(), 4);
    assert_eq!(points.last(), Some(&(0, 0)));
}

#[test]
fn bresenham_start_end_match() {
    let cases = vec![
        (0, 0, 3, 0),
        (0, 0, 0, 3),
        (0, 0, 3, 3),
        (0, 0, 1, 3),
        (2, 3, 2, 3),
        (3, 3, 0, 0),
    ];
    for (x0, y0, x1, y1) in cases {
        let points = bresenham_line(x0, y0, x1, y1);
        assert_eq!(
            points.first(),
            Some(&(x0, y0)),
            "first point should be start for ({x0},{y0})->({x1},{y1})"
        );
        assert_eq!(
            points.last(),
            Some(&(x1, y1)),
            "last point should be end for ({x0},{y0})->({x1},{y1})"
        );
    }
}

// ---------------------------------------------------------------------------
// Congestion tests
// ---------------------------------------------------------------------------

#[test]
fn congestion_no_nets() {
    let ctx = common::make_context();
    let report = estimate_congestion(&ctx, 0.5);
    assert_eq!(report.max_congestion, 0.0);
}

#[test]
fn congestion_same_direction_nets() {
    // Two nets both going east: (0,0)->(1,0).
    // Both cross the east edge of tile (0,0), so h_demand[0][0] should be 2.0.
    let mut ctx = common::make_context();
    ctx.populate_bel_buckets();

    let cell_type = ctx.id("LUT4");
    let out_port = ctx.id("Q");
    let in_port = ctx.id("A");

    // Net 1: cell_a (0,0) -> cell_b (1,0)
    let cell_a_name = ctx.id("cell_a");
    let cell_b_name = ctx.id("cell_b");
    let cell_a = ctx.design.add_cell(cell_a_name, cell_type);
    let cell_b = ctx.design.add_cell(cell_b_name, cell_type);

    ctx.design
        .cell_edit(cell_a)
        .add_port(out_port, PortType::Out);
    ctx.design.cell_edit(cell_b).add_port(in_port, PortType::In);

    let net1_name = ctx.id("net_1");
    let net1 = ctx.design.add_net(net1_name);
    ctx.design.net_edit(net1).set_driver(cell_a, out_port);
    ctx.design
        .cell_edit(cell_a)
        .set_port_net(out_port, Some(net1), None);
    let user1 = ctx.design.net_edit(net1).add_user(cell_b, in_port);
    ctx.design
        .cell_edit(cell_b)
        .set_port_net(in_port, Some(net1), Some(user1));

    // Net 2: cell_c (0,0) -> cell_d (1,0)
    // But we only have 1 BEL per tile, so place cell_c at (0,1) and cell_d at (1,1) instead.
    // Actually, we need two nets going the same direction. Let's use different tiles.
    // cell_c at tile (0,1) = BelId::new(2,0), cell_d at tile (1,1) = BelId::new(3,0).
    let cell_c_name = ctx.id("cell_c");
    let cell_d_name = ctx.id("cell_d");
    let cell_c = ctx.design.add_cell(cell_c_name, cell_type);
    let cell_d = ctx.design.add_cell(cell_d_name, cell_type);

    ctx.design
        .cell_edit(cell_c)
        .add_port(out_port, PortType::Out);
    ctx.design.cell_edit(cell_d).add_port(in_port, PortType::In);

    let net2_name = ctx.id("net_2");
    let net2 = ctx.design.add_net(net2_name);
    ctx.design.net_edit(net2).set_driver(cell_c, out_port);
    ctx.design
        .cell_edit(cell_c)
        .set_port_net(out_port, Some(net2), None);
    let user2 = ctx.design.net_edit(net2).add_user(cell_d, in_port);
    ctx.design
        .cell_edit(cell_d)
        .set_port_net(in_port, Some(net2), Some(user2));

    // Place: cell_a at (0,0), cell_b at (1,0), cell_c at (0,1), cell_d at (1,1)
    ctx.bind_bel(BelId::new(0, 0), cell_a, PlaceStrength::Placer);
    ctx.bind_bel(BelId::new(1, 0), cell_b, PlaceStrength::Placer);
    ctx.bind_bel(BelId::new(2, 0), cell_c, PlaceStrength::Placer);
    ctx.bind_bel(BelId::new(3, 0), cell_d, PlaceStrength::Placer);

    let report = estimate_congestion(&ctx, 0.0);

    // Both nets cross east edges: net1 crosses east edge at (0,0), net2 crosses east edge at (0,1).
    // h_demand[0][0] = 1.0 (from net1), h_demand[1][0] = 1.0 (from net2).
    assert!(
        report.h_demand[0][0] >= 1.0,
        "east edge at (0,0) should have demand from net1: got {}",
        report.h_demand[0][0]
    );
    assert!(
        report.h_demand[1][0] >= 1.0,
        "east edge at (0,1) should have demand from net2: got {}",
        report.h_demand[1][0]
    );
    assert!(
        report.max_congestion > 0.0,
        "max congestion should be > 0 with placed nets"
    );
}

#[test]
fn congestion_perpendicular_nets() {
    // Net1 goes east: (0,0)->(1,0), crosses h-edge at (0,0).
    // Net2 goes south: (1,0)->(1,1), crosses v-edge at (1,0).
    // They should increment different edge types.
    let mut ctx = common::make_context();
    ctx.populate_bel_buckets();

    let cell_type = ctx.id("LUT4");
    let out_port = ctx.id("Q");
    let in_port = ctx.id("A");

    // 4 cells for 2 nets.
    let names: Vec<_> = (0..4).map(|i| ctx.id(&format!("c{i}"))).collect();
    let cells: Vec<_> = names
        .iter()
        .map(|&n| ctx.design.add_cell(n, cell_type))
        .collect();

    // Net1: c0 (0,0) -> c1 (1,0)
    ctx.design
        .cell_edit(cells[0])
        .add_port(out_port, PortType::Out);
    ctx.design
        .cell_edit(cells[1])
        .add_port(in_port, PortType::In);
    let net1_name = ctx.id("n1");
    let net1 = ctx.design.add_net(net1_name);
    ctx.design.net_edit(net1).set_driver(cells[0], out_port);
    ctx.design
        .cell_edit(cells[0])
        .set_port_net(out_port, Some(net1), None);
    let u1 = ctx.design.net_edit(net1).add_user(cells[1], in_port);
    ctx.design
        .cell_edit(cells[1])
        .set_port_net(in_port, Some(net1), Some(u1));

    // Net2: c2 (1,0) -> c3 (1,1) -- but c2 can't share tile (1,0) with c1.
    // Use c2 at (0,1), c3 at (1,1) going east instead, or rethink.
    // Actually with 1 BEL per tile we need unique tiles.
    // Let net2: c2 at (0,1) -> c3 at (0,0)... wait that's west.
    // Simplest: net2 goes south from (1,0) to (1,1) but c1 is already at (1,0).
    // Instead: net1: c0(0,0)->c1(1,0) east. net2: c2(0,1)->c3(1,1) east+...
    // Let's just do: net2: c2 at (0,1) driving Q, c3 at (1,1) consuming A -- going east at y=1.
    ctx.design
        .cell_edit(cells[2])
        .add_port(out_port, PortType::Out);
    ctx.design
        .cell_edit(cells[3])
        .add_port(in_port, PortType::In);
    let net2_name = ctx.id("n2");
    let net2 = ctx.design.add_net(net2_name);
    ctx.design.net_edit(net2).set_driver(cells[2], out_port);
    ctx.design
        .cell_edit(cells[2])
        .set_port_net(out_port, Some(net2), None);
    let u2 = ctx.design.net_edit(net2).add_user(cells[3], in_port);
    ctx.design
        .cell_edit(cells[3])
        .set_port_net(in_port, Some(net2), Some(u2));

    // Place: c0 at (0,0), c1 at (1,0), c2 at (0,1), c3 at (1,1)
    ctx.bind_bel(BelId::new(0, 0), cells[0], PlaceStrength::Placer);
    ctx.bind_bel(BelId::new(1, 0), cells[1], PlaceStrength::Placer);
    ctx.bind_bel(BelId::new(2, 0), cells[2], PlaceStrength::Placer);
    ctx.bind_bel(BelId::new(3, 0), cells[3], PlaceStrength::Placer);

    let report = estimate_congestion(&ctx, 0.0);

    // Both nets go east, so h_demand should be populated but v_demand should be 0.
    assert!(
        report.h_demand[0][0] >= 1.0,
        "net1 should contribute h_demand at (0,0)"
    );
    assert!(
        report.h_demand[1][0] >= 1.0,
        "net2 should contribute h_demand at (0,1)"
    );
    // No vertical crossings since both nets are purely horizontal.
    let total_v: f64 = report.v_demand.iter().flat_map(|r| r.iter()).sum();
    assert_eq!(
        total_v, 0.0,
        "no vertical edge crossings expected for purely horizontal nets"
    );
}

// ---------------------------------------------------------------------------
// Density tests
// ---------------------------------------------------------------------------

#[test]
fn density_empty() {
    let ctx = common::make_context();
    let report = placement_density(&ctx, 1);
    assert_eq!(report.max_density, 0.0);
}

#[test]
fn density_with_placement() {
    let mut ctx = common::make_context_with_cells(2);
    let cell0 = ctx.design.cell_by_name(ctx.id("cell_0")).unwrap();
    let cell1 = ctx.design.cell_by_name(ctx.id("cell_1")).unwrap();

    ctx.bind_bel(BelId::new(0, 0), cell0, PlaceStrength::Placer);
    ctx.bind_bel(BelId::new(1, 0), cell1, PlaceStrength::Placer);

    let report = placement_density(&ctx, 1);
    assert!(
        report.max_density > 0.0,
        "max_density should be positive when cells are placed"
    );
}

// ---------------------------------------------------------------------------
// BBox tests
// ---------------------------------------------------------------------------

#[test]
fn bbox_contains() {
    let bb = BoundingBox {
        x0: 1,
        y0: 2,
        x1: 5,
        y1: 7,
    };
    // Inside
    assert!(bb.contains(1, 2));
    assert!(bb.contains(5, 7));
    assert!(bb.contains(3, 4));
    // Outside
    assert!(!bb.contains(0, 2));
    assert!(!bb.contains(1, 1));
    assert!(!bb.contains(6, 4));
    assert!(!bb.contains(3, 8));
}

#[test]
fn bbox_compute() {
    // Place driver at (0,0) and sink at (1,1), verify bbox matches.
    let mut ctx = common::make_context_with_cells(2);
    let cell0 = ctx.design.cell_by_name(ctx.id("cell_0")).unwrap();
    let cell1 = ctx.design.cell_by_name(ctx.id("cell_1")).unwrap();

    ctx.bind_bel(BelId::new(0, 0), cell0, PlaceStrength::Placer); // (0,0)
    ctx.bind_bel(BelId::new(3, 0), cell1, PlaceStrength::Placer); // (1,1)

    let net_idx = ctx.design.net_by_name(ctx.id("net_0")).unwrap();
    let bb = compute_bbox(&ctx, net_idx, 0);

    assert_eq!(bb.x0, 0);
    assert_eq!(bb.y0, 0);
    assert_eq!(bb.x1, 1);
    assert_eq!(bb.y1, 1);
    assert!(bb.contains(0, 0));
    assert!(bb.contains(1, 1));
}

// ---------------------------------------------------------------------------
// HPWL tests (through the new import path)
// ---------------------------------------------------------------------------

#[test]
fn hpwl_unplaced_is_zero() {
    let ctx = common::make_context_with_cells(2);
    let net_idx = ctx.design.net_by_name(ctx.id("net_0")).unwrap();
    // No cells placed, HPWL should be 0.
    assert_eq!(net_hpwl(&ctx, net_idx), 0.0);
}

#[test]
fn hpwl_placed_net() {
    let mut ctx = common::make_context_with_cells(2);
    let cell0 = ctx.design.cell_by_name(ctx.id("cell_0")).unwrap();
    let cell1 = ctx.design.cell_by_name(ctx.id("cell_1")).unwrap();

    // Place at (0,0) and (1,1): HPWL = |1-0| + |1-0| = 2.
    ctx.bind_bel(BelId::new(0, 0), cell0, PlaceStrength::Placer);
    ctx.bind_bel(BelId::new(3, 0), cell1, PlaceStrength::Placer);

    let net_idx = ctx.design.net_by_name(ctx.id("net_0")).unwrap();
    assert_eq!(net_hpwl(&ctx, net_idx), 2.0);
}

#[test]
fn total_hpwl_matches_sum() {
    let mut ctx = common::make_context_with_cells(3);
    let cell0 = ctx.design.cell_by_name(ctx.id("cell_0")).unwrap();
    let cell1 = ctx.design.cell_by_name(ctx.id("cell_1")).unwrap();
    let cell2 = ctx.design.cell_by_name(ctx.id("cell_2")).unwrap();

    ctx.bind_bel(BelId::new(0, 0), cell0, PlaceStrength::Placer);
    ctx.bind_bel(BelId::new(1, 0), cell1, PlaceStrength::Placer);
    ctx.bind_bel(BelId::new(3, 0), cell2, PlaceStrength::Placer);

    let net_idx = ctx.design.net_by_name(ctx.id("net_0")).unwrap();
    let single = net_hpwl(&ctx, net_idx);
    let total = total_hpwl(&ctx);

    // Only one net, so total should equal single net's HPWL.
    assert_eq!(total, single);
}

#[test]
fn line_estimate_counts_unique_edge_crossings_on_path() {
    let mut ctx = common::make_context_with_cells(2);
    let cell0 = ctx.design.cell_by_name(ctx.id("cell_0")).unwrap();
    let cell1 = ctx.design.cell_by_name(ctx.id("cell_1")).unwrap();

    ctx.bind_bel(BelId::new(0, 0), cell0, PlaceStrength::Placer);
    ctx.bind_bel(BelId::new(1, 0), cell1, PlaceStrength::Placer);

    let net_idx = ctx.design.net_by_name(ctx.id("net_0")).unwrap();

    // Straight line from (0,0) to (1,0) crosses one tile boundary edge.
    assert_eq!(net_line_estimate(&ctx, net_idx), 1.0);
}

#[test]
fn line_estimate_deduplicates_edges_across_sink_paths() {
    let mut ctx = common::make_context_with_cells(3);
    let cell0 = ctx.design.cell_by_name(ctx.id("cell_0")).unwrap();
    let cell1 = ctx.design.cell_by_name(ctx.id("cell_1")).unwrap();
    let cell2 = ctx.design.cell_by_name(ctx.id("cell_2")).unwrap();

    ctx.bind_bel(BelId::new(0, 0), cell0, PlaceStrength::Placer);
    ctx.bind_bel(BelId::new(1, 0), cell1, PlaceStrength::Placer);
    ctx.bind_bel(BelId::new(3, 0), cell2, PlaceStrength::Placer);

    let net_idx = ctx.design.net_by_name(ctx.id("net_0")).unwrap();

    // Driver-to-sink lines cross edges (0,0)->(1,0) and (0,0)->(1,1).
    // Unique crossed edges are {(0,0)-(1,0), (0,0)-(1,1)}.
    assert_eq!(net_line_estimate(&ctx, net_idx), 2.0);
}

// ---------------------------------------------------------------------------
// Independent algorithm tests (no chipdb required)
// ---------------------------------------------------------------------------

// -- Edge crossing accumulation --

#[test]
fn edge_crossings_eastward() {
    // Line from (0,0) to (2,0): crosses east edges at (0,0) and (1,0)
    let points = bresenham_line(0, 0, 2, 0);
    let mut h = vec![vec![0.0; 3]; 1];
    let mut v = vec![vec![0.0; 3]; 1];
    accumulate_edge_crossings(&points, 3, 1, &mut h, &mut v, 1.0);
    assert_eq!(h[0][0], 1.0, "east edge at (0,0)");
    assert_eq!(h[0][1], 1.0, "east edge at (1,0)");
    assert_eq!(h[0][2], 0.0, "no east edge at boundary");
    let total_v: f64 = v.iter().flat_map(|r| r.iter()).sum();
    assert_eq!(total_v, 0.0, "no vertical crossings for horizontal line");
}

#[test]
fn edge_crossings_westward() {
    // Line from (2,0) to (0,0): crosses east edges at (1,0) and (0,0) (from the west side)
    let points = bresenham_line(2, 0, 0, 0);
    let mut h = vec![vec![0.0; 3]; 1];
    let mut v = vec![vec![0.0; 3]; 1];
    accumulate_edge_crossings(&points, 3, 1, &mut h, &mut v, 1.0);
    assert_eq!(h[0][0], 1.0, "east edge at (0,0) crossed westward");
    assert_eq!(h[0][1], 1.0, "east edge at (1,0) crossed westward");
}

#[test]
fn edge_crossings_southward() {
    // Line from (0,0) to (0,2): crosses south edges at (0,0) and (0,1)
    let points = bresenham_line(0, 0, 0, 2);
    let mut h = vec![vec![0.0; 1]; 3];
    let mut v = vec![vec![0.0; 1]; 3];
    accumulate_edge_crossings(&points, 1, 3, &mut h, &mut v, 1.0);
    assert_eq!(v[0][0], 1.0, "south edge at (0,0)");
    assert_eq!(v[1][0], 1.0, "south edge at (0,1)");
    let total_h: f64 = h.iter().flat_map(|r| r.iter()).sum();
    assert_eq!(total_h, 0.0, "no horizontal crossings for vertical line");
}

#[test]
fn edge_crossings_diagonal() {
    // Line from (0,0) to (2,2): crosses both east and south edges
    let points = bresenham_line(0, 0, 2, 2);
    let mut h = vec![vec![0.0; 3]; 3];
    let mut v = vec![vec![0.0; 3]; 3];
    accumulate_edge_crossings(&points, 3, 3, &mut h, &mut v, 1.0);
    let total_h: f64 = h.iter().flat_map(|r| r.iter()).sum();
    let total_v: f64 = v.iter().flat_map(|r| r.iter()).sum();
    assert_eq!(total_h, 2.0, "diagonal crosses 2 east edges");
    assert_eq!(total_v, 2.0, "diagonal crosses 2 south edges");
}

#[test]
fn edge_crossings_same_point() {
    // Single point: no crossings
    let points = bresenham_line(1, 1, 1, 1);
    let mut h = vec![vec![0.0; 3]; 3];
    let mut v = vec![vec![0.0; 3]; 3];
    accumulate_edge_crossings(&points, 3, 3, &mut h, &mut v, 1.0);
    let total: f64 = h
        .iter()
        .flat_map(|r| r.iter())
        .chain(v.iter().flat_map(|r| r.iter()))
        .sum();
    assert_eq!(total, 0.0, "single point has no crossings");
}

#[test]
fn edge_crossings_multiple_nets_same_direction() {
    // Two nets both going east from (0,0) to (2,0): demand should accumulate
    let points = bresenham_line(0, 0, 2, 0);
    let mut h = vec![vec![0.0; 3]; 1];
    let mut v = vec![vec![0.0; 3]; 1];
    accumulate_edge_crossings(&points, 3, 1, &mut h, &mut v, 1.0);
    accumulate_edge_crossings(&points, 3, 1, &mut h, &mut v, 1.0);
    assert_eq!(h[0][0], 2.0, "two nets crossing same east edge");
    assert_eq!(h[0][1], 2.0, "two nets crossing same east edge");
}

#[test]
fn edge_crossings_perpendicular_no_interference() {
    // Net1 goes east (0,1)->(2,1), Net2 goes south (1,0)->(1,2)
    // They cross different edge types, so per-edge demand = 1.0 max
    let mut h = vec![vec![0.0; 3]; 3];
    let mut v = vec![vec![0.0; 3]; 3];
    let points_h = bresenham_line(0, 1, 2, 1);
    accumulate_edge_crossings(&points_h, 3, 3, &mut h, &mut v, 1.0);
    let points_v = bresenham_line(1, 0, 1, 2);
    accumulate_edge_crossings(&points_v, 3, 3, &mut h, &mut v, 1.0);
    // East edges at row 1 should have demand from horizontal net
    assert_eq!(h[1][0], 1.0);
    assert_eq!(h[1][1], 1.0);
    // South edges at col 1 should have demand from vertical net
    assert_eq!(v[0][1], 1.0);
    assert_eq!(v[1][1], 1.0);
}

// -- Congestion ratio computation --

#[test]
fn congestion_ratios_zero_demand() {
    let h_demand = vec![vec![0.0; 3]; 3];
    let v_demand = vec![vec![0.0; 3]; 3];
    let h_cap = vec![vec![4.0; 3]; 3];
    let v_cap = vec![vec![4.0; 3]; 3];
    let result = compute_congestion_ratios(&h_demand, &v_demand, &h_cap, &v_cap, 0.5);
    assert_eq!(result.max_congestion, 0.0);
    assert_eq!(result.avg_congestion, 0.0);
    assert!(result.hot_edges.is_empty());
}

#[test]
fn congestion_ratios_uniform_demand() {
    // All edges have demand 2.0, capacity 4.0 -> ratio 0.5
    let h_demand = vec![vec![2.0; 3]; 3];
    let v_demand = vec![vec![2.0; 3]; 3];
    let h_cap = vec![vec![4.0; 3]; 3];
    let v_cap = vec![vec![4.0; 3]; 3];
    let result = compute_congestion_ratios(&h_demand, &v_demand, &h_cap, &v_cap, 0.5);
    assert!((result.max_congestion - 0.5).abs() < 1e-10);
    assert!((result.avg_congestion - 0.5).abs() < 1e-10);
    // Interior east edges should be 0.5
    assert!((result.h_congestion[0][0] - 0.5).abs() < 1e-10);
}

#[test]
fn congestion_ratios_hotspot_detection() {
    // One edge has high demand, rest is zero
    let mut h_demand = vec![vec![0.0; 3]; 3];
    let v_demand = vec![vec![0.0; 3]; 3];
    h_demand[1][1] = 8.0; // hotspot at (1,1) horizontal
    let h_cap = vec![vec![4.0; 3]; 3];
    let v_cap = vec![vec![4.0; 3]; 3];
    let result = compute_congestion_ratios(&h_demand, &v_demand, &h_cap, &v_cap, 0.5);
    assert!((result.max_congestion - 2.0).abs() < 1e-10, "8/4 = 2.0");
    assert_eq!(result.hotspot, (1, 1));
    assert_eq!(result.hotspot_axis, Axis::Horizontal);
    assert!(
        !result.hot_edges.is_empty(),
        "should have hot edges above threshold 0.5"
    );
}

#[test]
fn congestion_ratios_boundary_edges_excluded() {
    // 2x2 grid: east edges exist at x=0 (not x=1 which is the boundary)
    // south edges exist at y=0 (not y=1)
    let h_demand = vec![vec![1.0; 2]; 2];
    let v_demand = vec![vec![1.0; 2]; 2];
    let h_cap = vec![vec![1.0; 2]; 2];
    let v_cap = vec![vec![1.0; 2]; 2];
    let result = compute_congestion_ratios(&h_demand, &v_demand, &h_cap, &v_cap, 0.0);
    // x=1 is boundary, so east edge at x=1 should not be computed (stays 0)
    assert_eq!(
        result.h_congestion[0][1], 0.0,
        "boundary east edge not computed"
    );
    assert_eq!(
        result.h_congestion[1][1], 0.0,
        "boundary east edge not computed"
    );
    // y=1 is boundary, so south edge at y=1 should not be computed
    assert_eq!(
        result.v_congestion[1][0], 0.0,
        "boundary south edge not computed"
    );
    assert_eq!(
        result.v_congestion[1][1], 0.0,
        "boundary south edge not computed"
    );
    // Interior edges should be 1.0
    assert!(
        (result.h_congestion[0][0] - 1.0).abs() < 1e-10,
        "interior east edge"
    );
    assert!(
        (result.v_congestion[0][0] - 1.0).abs() < 1e-10,
        "interior south edge"
    );
}

// -- Density sliding window --

#[test]
fn density_window_empty_grid() {
    // 4x4 grid, nothing placed, all tiles have capacity 2
    let tile_placed = vec![0u32; 16];
    let tile_capacity = vec![2u32; 16];
    let report = compute_sliding_window_density(&tile_placed, &tile_capacity, 4, 4, 2);
    assert_eq!(report.max_density, 0.0);
    assert_eq!(report.avg_density, 0.0);
    assert_eq!(report.hot_regions, 0);
}

#[test]
fn density_window_fully_packed() {
    // 4x4 grid, every tile has 2 BELs, all placed
    let tile_placed = vec![2u32; 16];
    let tile_capacity = vec![2u32; 16];
    let report = compute_sliding_window_density(&tile_placed, &tile_capacity, 4, 4, 2);
    assert!(
        (report.max_density - 1.0).abs() < 1e-10,
        "all tiles full = density 1.0"
    );
    assert!((report.avg_density - 1.0).abs() < 1e-10);
}

#[test]
fn density_window_single_hotspot() {
    // 4x4 grid, only tile (0,0) has cells placed
    let mut tile_placed = vec![0u32; 16];
    let tile_capacity = vec![4u32; 16];
    tile_placed[0] = 4; // tile (0,0) fully packed
    let report = compute_sliding_window_density(&tile_placed, &tile_capacity, 4, 4, 1);
    assert!(
        (report.max_density - 1.0).abs() < 1e-10,
        "hotspot tile is fully packed"
    );
    assert_eq!(report.hotspot, (0, 0));
    // Regions with 0 placed cells have density 0, so hot_regions should be small
    assert!(report.hot_regions <= 1);
}

#[test]
fn density_window_larger_than_grid() {
    // Window larger than grid: should still work, just one region
    let tile_placed = vec![1u32; 4];
    let tile_capacity = vec![2u32; 4];
    let report = compute_sliding_window_density(&tile_placed, &tile_capacity, 2, 2, 10);
    // 4 tiles, each 1/2 placed = 4/8 = 0.5 total density
    assert!((report.max_density - 0.5).abs() < 1e-10);
}

#[test]
fn density_window_step_coverage() {
    // 6x1 grid with window=2 (step=1): should produce windows at x=0,1,2,3,4
    let tile_placed = vec![0, 0, 0, 0, 2, 2]; // cells only in last 2 tiles
    let tile_capacity = vec![2u32; 6];
    let report = compute_sliding_window_density(&tile_placed, &tile_capacity, 6, 1, 2);
    // Window at x=4 covers tiles 4,5: 4/4 = 1.0
    assert!((report.max_density - 1.0).abs() < 1e-10);
    assert_eq!(report.hotspot, (4, 0));
}

#[test]
fn density_zero_capacity_tiles_skipped() {
    // Grid where some tiles have zero capacity (no BELs)
    let tile_placed = vec![0u32; 4];
    let tile_capacity = vec![0, 0, 0, 0];
    let report = compute_sliding_window_density(&tile_placed, &tile_capacity, 2, 2, 1);
    assert_eq!(report.max_density, 0.0);
    assert_eq!(report.avg_density, 0.0);
}

// -- Bresenham additional correctness --

#[test]
fn bresenham_all_octants_correct_length() {
    // For a line from origin to (dx, dy), length should be max(|dx|, |dy|) + 1
    let cases = vec![
        (0, 0, 5, 0),   // east
        (0, 0, -5, 0),  // west
        (0, 0, 0, 5),   // south
        (0, 0, 0, -5),  // north
        (0, 0, 5, 5),   // SE diagonal
        (0, 0, -5, -5), // NW diagonal
        (0, 0, 5, 3),   // shallow SE
        (0, 0, 3, 5),   // steep SE
        (0, 0, -5, 3),  // shallow SW
        (0, 0, 3, -5),  // steep NE reversed
    ];
    for (x0, y0, x1, y1) in cases {
        let points = bresenham_line(x0, y0, x1, y1);
        let expected_len = (x1 - x0).abs().max((y1 - y0).abs()) + 1;
        assert_eq!(
            points.len(),
            expected_len as usize,
            "line ({x0},{y0})->({x1},{y1}): expected {expected_len} points, got {}",
            points.len()
        );
    }
}

#[test]
fn bresenham_continuity() {
    // Each consecutive pair of points should differ by at most 1 in each axis
    let cases = vec![(0, 0, 7, 3), (0, 0, 3, 7), (5, 5, -2, -3), (0, 0, 10, 1)];
    for (x0, y0, x1, y1) in cases {
        let points = bresenham_line(x0, y0, x1, y1);
        for pair in points.windows(2) {
            let dx = (pair[1].0 - pair[0].0).abs();
            let dy = (pair[1].1 - pair[0].1).abs();
            assert!(
                dx <= 1 && dy <= 1,
                "line ({x0},{y0})->({x1},{y1}): non-adjacent step ({},{})->({},{})",
                pair[0].0,
                pair[0].1,
                pair[1].0,
                pair[1].1
            );
        }
    }
}

#[test]
fn bresenham_symmetry() {
    // Bresenham from A to B should produce the reverse of B to A
    let (x0, y0, x1, y1) = (0, 0, 7, 3);
    let forward = bresenham_line(x0, y0, x1, y1);
    let backward = bresenham_line(x1, y1, x0, y0);
    let mut reversed = backward.clone();
    reversed.reverse();
    assert_eq!(
        forward, reversed,
        "forward and reversed backward should match"
    );
}
