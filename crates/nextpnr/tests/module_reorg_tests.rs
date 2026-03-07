use nextpnr::chipdb::{BelId, Loc, PipId, WireId};
use nextpnr::common::{IdString, IdStringPool, IntoIdString, PlaceStrength};
use nextpnr::netlist::{PortType, Property};
use nextpnr::timing::{ClockEdge, DelayPair, DelayQuad, DelayT, TimingPortClass};

#[test]
fn reorg_public_modules_expose_shared_types() {
    let pool = IdStringPool::new();
    let id = "clk".into_id(&pool);

    let bel = BelId::new(1, 2);
    let wire = WireId::new(3, 4);
    let pip = PipId::new(5, 6);
    let loc = Loc::new(7, 8, 9);
    let prop = Property::string("value");
    let delay: DelayT = 42;
    let pair = DelayPair::uniform(delay);
    let quad = DelayQuad::uniform(delay);

    assert_eq!(pool.lookup(id), Some("clk"));
    assert_eq!(id, IdString(1));
    assert!(bel.is_valid());
    assert!(wire.is_valid());
    assert!(pip.is_valid());
    assert_eq!(loc.x, 7);
    assert_eq!(PortType::InOut.to_string(), "INOUT");
    assert_eq!(TimingPortClass::ClockInput.to_string(), "CLOCK_INPUT");
    assert_eq!(ClockEdge::Rising.opposite(), ClockEdge::Falling);
    assert_eq!(pair.average(), 42);
    assert_eq!(quad.max_delay(), 42);
    assert!(matches!(prop, Property::String(_)));
    assert!(PlaceStrength::Fixed.is_locked());
}
