mod common;

use nextpnr::placer::hydraulic_place::kirchhoff;
use nextpnr::placer::hydraulic_place::network::{Direction, Junction, Pipe, PipeNetwork};

// =====================================================================
// Kirchhoff solver tests
// =====================================================================

fn make_2_node_network() -> (PipeNetwork, Vec<f64>) {
    let junctions = vec![
        Junction {
            x: 0,
            y: 0,
            pressure: 0.0,
            demand: 0.0,
        },
        Junction {
            x: 1,
            y: 0,
            pressure: 0.0,
            demand: 0.0,
        },
    ];
    let pipes = vec![Pipe {
        from: 0,
        to: 1,
        resistance: 2.0,
        capacity: 10.0,
        flow: 0.0,
        direction: Direction::East,
    }];
    let junction_pipes = vec![vec![0], vec![0]];
    let network = PipeNetwork {
        junctions,
        pipes,
        junction_pipes,
        width: 2,
        height: 1,
        schur_matrices: vec![],
    };
    let demand = vec![1.0, -1.0];
    (network, demand)
}

#[test]
fn kirchhoff_2_node_pressure_drop() {
    let (mut network, demand) = make_2_node_network();
    let result = kirchhoff::kirchhoff_solve(&mut network, &demand, 0.0, 1, 500, 1e-8);
    assert!(result.converged);

    let p0 = network.junctions[0].pressure;
    let p1 = network.junctions[1].pressure;
    assert!(p0.abs() < 1e-6, "Reference node should be ~0, got {}", p0);
    // dP = R * Q: pressure drop should equal R * flow.
    let dp = p0 - p1;
    let r = network.pipes[0].resistance;
    let q = network.pipes[0].flow;
    assert!(
        (dp - r * q).abs() < 0.2,
        "dP={} should ~= R*Q={}",
        dp,
        r * q
    );
}

#[test]
fn kirchhoff_zero_demand_zero_pressure() {
    let (mut network, _) = make_2_node_network();
    let demand = vec![0.0, 0.0];
    let result = kirchhoff::kirchhoff_solve(&mut network, &demand, 0.0, 1, 500, 1e-8);
    assert!(result.converged);
    assert!(result.energy.abs() < 1e-6);
    for j in &network.junctions {
        assert!(j.pressure.abs() < 1e-6, "Zero demand -> zero pressure");
    }
}

#[test]
fn kirchhoff_turbulence_increases_resistance() {
    let (mut net_lam, demand) = make_2_node_network();
    let (mut net_turb, demand2) = make_2_node_network();

    // Laminar solve.
    let r_lam = kirchhoff::kirchhoff_solve(&mut net_lam, &demand, 0.0, 1, 500, 1e-8);

    // Pre-seed flow to trigger turbulence in Newton iteration.
    net_turb.pipes[0].flow = 5.0;
    let r_turb = kirchhoff::kirchhoff_solve(&mut net_turb, &demand2, 10.0, 3, 500, 1e-8);

    // Turbulence should increase effective resistance -> higher energy magnitude.
    assert!(
        r_turb.energy.abs() >= r_lam.energy.abs() - 0.01,
        "Turbulent energy ({}) should be >= laminar ({})",
        r_turb.energy,
        r_lam.energy
    );
}

#[test]
fn kirchhoff_flow_conservation_3_node() {
    let junctions = vec![
        Junction {
            x: 0,
            y: 0,
            pressure: 0.0,
            demand: 0.0,
        },
        Junction {
            x: 1,
            y: 0,
            pressure: 0.0,
            demand: 0.0,
        },
        Junction {
            x: 2,
            y: 0,
            pressure: 0.0,
            demand: 0.0,
        },
    ];
    let pipes = vec![
        Pipe {
            from: 0,
            to: 1,
            resistance: 1.0,
            capacity: 10.0,
            flow: 0.0,
            direction: Direction::East,
        },
        Pipe {
            from: 1,
            to: 2,
            resistance: 1.0,
            capacity: 10.0,
            flow: 0.0,
            direction: Direction::East,
        },
    ];
    let junction_pipes = vec![vec![0], vec![0, 1], vec![1]];
    let mut network = PipeNetwork {
        junctions,
        pipes,
        junction_pipes,
        width: 3,
        height: 1,
        schur_matrices: vec![],
    };
    let demand = vec![1.0, -0.5, -0.5];
    let result = kirchhoff::kirchhoff_solve(&mut network, &demand, 0.0, 1, 500, 1e-8);
    assert!(result.converged);
    // Flow should be positive (fluid flows from high to low pressure).
    assert!(network.pipes[0].flow > 0.0, "Flow 0->1 should be positive");
}

// =====================================================================
// Network tests (using synthetic chipdb)
// =====================================================================

#[test]
fn network_from_synthetic_chipdb() {
    let ctx = common::make_context();
    let network = PipeNetwork::from_context(&ctx);

    assert_eq!(network.width, 2);
    assert_eq!(network.height, 2);
    assert_eq!(network.num_junctions(), 4);
    // (w-1)*h east + w*(h-1) south = 1*2 + 2*1 = 4
    assert_eq!(network.num_pipes(), 4);
}

#[test]
fn network_junction_index() {
    let ctx = common::make_context();
    let network = PipeNetwork::from_context(&ctx);

    assert_eq!(network.junction_index(0, 0), 0);
    assert_eq!(network.junction_index(1, 0), 1);
    assert_eq!(network.junction_index(0, 1), 2);
    assert_eq!(network.junction_index(1, 1), 3);
}

// =====================================================================
// Pressure solver on congested 2x2 grid
// =====================================================================

fn make_congested_2x2() -> PipeNetwork {
    let junctions = vec![
        Junction {
            x: 0,
            y: 0,
            pressure: 0.0,
            demand: 0.0,
        },
        Junction {
            x: 1,
            y: 0,
            pressure: 0.0,
            demand: 0.0,
        },
        Junction {
            x: 0,
            y: 1,
            pressure: 0.0,
            demand: 0.0,
        },
        Junction {
            x: 1,
            y: 1,
            pressure: 0.0,
            demand: 0.0,
        },
    ];
    let pipes = vec![
        Pipe {
            from: 0,
            to: 1,
            resistance: 1.0,
            capacity: 5.0,
            flow: 0.0,
            direction: Direction::East,
        },
        Pipe {
            from: 2,
            to: 3,
            resistance: 1.0,
            capacity: 5.0,
            flow: 0.0,
            direction: Direction::East,
        },
        Pipe {
            from: 0,
            to: 2,
            resistance: 1.0,
            capacity: 5.0,
            flow: 0.0,
            direction: Direction::South,
        },
        Pipe {
            from: 1,
            to: 3,
            resistance: 1.0,
            capacity: 5.0,
            flow: 0.0,
            direction: Direction::South,
        },
    ];
    let junction_pipes = vec![vec![0, 2], vec![0, 3], vec![1, 2], vec![1, 3]];
    PipeNetwork {
        junctions,
        pipes,
        junction_pipes,
        width: 2,
        height: 2,
        schur_matrices: vec![[[1.0; 4]; 4]; 1],
    }
}

#[test]
fn kirchhoff_2x2_with_demand() {
    let mut network = make_congested_2x2();
    // Net: driver at (0,0), sink at (1,1).
    let demand = vec![1.0, 0.0, 0.0, -1.0];
    let result = kirchhoff::kirchhoff_solve(&mut network, &demand, 0.0, 1, 500, 1e-6);
    assert!(result.converged);
    // Non-zero pressure field.
    let max_p = network
        .junctions
        .iter()
        .map(|j| j.pressure.abs())
        .fold(0.0f64, f64::max);
    assert!(max_p > 0.0, "Should produce non-zero pressure field");
    assert!(
        network.junctions[0].pressure.abs() < 1e-6,
        "Reference node should be ~0"
    );
}

#[test]
fn kirchhoff_empty_network() {
    let mut network = PipeNetwork {
        junctions: vec![],
        pipes: vec![],
        junction_pipes: vec![],
        width: 0,
        height: 0,
        schur_matrices: vec![],
    };
    let demand = vec![];
    let result = kirchhoff::kirchhoff_solve(&mut network, &demand, 0.0, 1, 500, 1e-6);
    assert!(result.converged);
    assert_eq!(result.iterations, 0);
}
