use nextpnr::chipdb::{BelId, Loc, PipId, WireId};
use nextpnr::common::{IdString, IdStringPool, PlaceStrength};
use nextpnr::netlist::{PortType, Property};
use nextpnr::timing::{ClockEdge, DelayPair, DelayQuad, TimingPortClass};

// ============================================================================
// Tests from ids.rs
// ============================================================================

macro_rules! test_packed_id {
    ($name:ident, $ty:ident) => {
        mod $name {
            use super::*;

            #[test]
            fn new_and_accessors() {
                let id = $ty::new(10, 20);
                assert_eq!(id.tile(), 10);
                assert_eq!(id.index(), 20);
                assert!(id.is_valid());
            }

            #[test]
            fn zero_values() {
                let id = $ty::new(0, 0);
                assert_eq!(id.tile(), 0);
                assert_eq!(id.index(), 0);
                assert!(id.is_valid());
            }

            #[test]
            fn negative_values() {
                let id = $ty::new(-5, -10);
                assert_eq!(id.tile(), -5);
                assert_eq!(id.index(), -10);
                // tile == -5 != -1, so it's valid
                assert!(id.is_valid());
            }

            #[test]
            fn invalid_constant() {
                let id = $ty::INVALID;
                assert_eq!(id.tile(), -1);
                assert_eq!(id.index(), 0);
                assert!(!id.is_valid());
            }

            #[test]
            fn default_is_zero() {
                let id = $ty::default();
                assert_eq!(id.tile(), 0);
                assert_eq!(id.index(), 0);
                assert!(id.is_valid());
            }

            #[test]
            fn equality() {
                let a = $ty::new(1, 2);
                let b = $ty::new(1, 2);
                let c = $ty::new(1, 3);
                assert_eq!(a, b);
                assert_ne!(a, c);
            }

            #[test]
            fn hashing() {
                use std::collections::HashSet;
                let mut set = HashSet::new();
                set.insert($ty::new(1, 2));
                set.insert($ty::new(3, 4));
                set.insert($ty::new(1, 2)); // duplicate
                assert_eq!(set.len(), 2);
            }

            #[test]
            fn copy_semantics() {
                let a = $ty::new(5, 6);
                let b = a;
                assert_eq!(a, b); // a is still usable after copy
            }

            #[test]
            fn max_values() {
                let id = $ty::new(i32::MAX, i32::MAX);
                assert_eq!(id.tile(), i32::MAX);
                assert_eq!(id.index(), i32::MAX);
                assert!(id.is_valid());
            }

            #[test]
            fn min_values() {
                let id = $ty::new(i32::MIN, i32::MIN);
                assert_eq!(id.tile(), i32::MIN);
                assert_eq!(id.index(), i32::MIN);
                // i32::MIN != -1, so it's valid
                assert!(id.is_valid());
            }

            #[test]
            fn raw_roundtrip() {
                let id = $ty::new(42, 99);
                let raw = id.raw();
                let restored = $ty::from_raw(raw);
                assert_eq!(id, restored);
                assert_eq!(restored.tile(), 42);
                assert_eq!(restored.index(), 99);
            }

            #[test]
            fn debug_format_valid() {
                let id = $ty::new(1, 2);
                let debug = format!("{:?}", id);
                assert!(debug.contains(stringify!($ty)));
                assert!(debug.contains("tile=1"));
                assert!(debug.contains("index=2"));
            }

            #[test]
            fn debug_format_invalid() {
                let id = $ty::INVALID;
                let debug = format!("{:?}", id);
                assert!(debug.contains("INVALID"));
            }

            #[test]
            fn display_format_valid() {
                let id = $ty::new(1, 2);
                assert_eq!(format!("{}", id), "1:2");
            }

            #[test]
            fn display_format_invalid() {
                let id = $ty::INVALID;
                assert_eq!(format!("{}", id), "<invalid>");
            }

            #[test]
            fn tile_negative_one_is_invalid() {
                // Any ID with tile == -1 is invalid, regardless of index
                let id = $ty::new(-1, 42);
                assert!(!id.is_valid());
            }
        }
    };
}

test_packed_id!(bel_id, BelId);
test_packed_id!(wire_id, WireId);
test_packed_id!(pip_id, PipId);

#[test]
fn different_types_are_distinct() {
    // Ensure BelId, WireId, PipId are not accidentally interchangeable at the type level.
    // This is a compile-time check -- if this compiles, the types are distinct.
    let _bel: BelId = BelId::new(0, 0);
    let _wire: WireId = WireId::new(0, 0);
    let _pip: PipId = PipId::new(0, 0);
}

// ============================================================================
// Tests from delay.rs
// ============================================================================

// === DelayPair tests ===

#[test]
fn delay_pair_new() {
    let dp = DelayPair::new(100, 200);
    assert_eq!(dp.min_delay, 100);
    assert_eq!(dp.max_delay, 200);
}

#[test]
fn delay_pair_uniform() {
    let dp = DelayPair::uniform(150);
    assert_eq!(dp.min_delay, 150);
    assert_eq!(dp.max_delay, 150);
}

#[test]
fn delay_pair_default() {
    let dp = DelayPair::default();
    assert_eq!(dp.min_delay, 0);
    assert_eq!(dp.max_delay, 0);
}

#[test]
fn delay_pair_average() {
    let dp = DelayPair::new(100, 200);
    assert_eq!(dp.average(), 150);
}

#[test]
fn delay_pair_average_uniform() {
    let dp = DelayPair::uniform(300);
    assert_eq!(dp.average(), 300);
}

#[test]
fn delay_pair_add() {
    let a = DelayPair::new(100, 200);
    let b = DelayPair::new(10, 20);
    let c = a + b;
    assert_eq!(c.min_delay, 110);
    assert_eq!(c.max_delay, 220);
}

#[test]
fn delay_pair_sub() {
    let a = DelayPair::new(100, 200);
    let b = DelayPair::new(10, 20);
    let c = a - b;
    assert_eq!(c.min_delay, 90);
    assert_eq!(c.max_delay, 180);
}

#[test]
fn delay_pair_equality() {
    let a = DelayPair::new(100, 200);
    let b = DelayPair::new(100, 200);
    let c = DelayPair::new(100, 300);
    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn delay_pair_copy() {
    let a = DelayPair::new(100, 200);
    let b = a;
    assert_eq!(a, b);
}

#[test]
fn delay_pair_debug() {
    let dp = DelayPair::new(100, 200);
    let s = format!("{:?}", dp);
    assert!(s.contains("min=100"));
    assert!(s.contains("max=200"));
}

#[test]
fn delay_pair_display_uniform() {
    let dp = DelayPair::uniform(100);
    assert_eq!(format!("{}", dp), "100ps");
}

#[test]
fn delay_pair_display_range() {
    let dp = DelayPair::new(100, 200);
    assert_eq!(format!("{}", dp), "100-200ps");
}

#[test]
fn delay_pair_negative() {
    let dp = DelayPair::new(-50, -10);
    assert_eq!(dp.min_delay, -50);
    assert_eq!(dp.max_delay, -10);
    assert_eq!(dp.average(), -30);
}

// === DelayQuad tests ===

#[test]
fn delay_quad_new() {
    let rise = DelayPair::new(100, 200);
    let fall = DelayPair::new(150, 250);
    let dq = DelayQuad::new(rise, fall);
    assert_eq!(dq.rise, rise);
    assert_eq!(dq.fall, fall);
}

#[test]
fn delay_quad_uniform() {
    let dq = DelayQuad::uniform(100);
    assert_eq!(dq.rise.min_delay, 100);
    assert_eq!(dq.rise.max_delay, 100);
    assert_eq!(dq.fall.min_delay, 100);
    assert_eq!(dq.fall.max_delay, 100);
}

#[test]
fn delay_quad_uniform_pair() {
    let pair = DelayPair::new(100, 200);
    let dq = DelayQuad::uniform_pair(pair);
    assert_eq!(dq.rise, pair);
    assert_eq!(dq.fall, pair);
}

#[test]
fn delay_quad_default() {
    let dq = DelayQuad::default();
    assert_eq!(dq.rise, DelayPair::default());
    assert_eq!(dq.fall, DelayPair::default());
}

#[test]
fn delay_quad_min_delay() {
    let dq = DelayQuad::new(
        DelayPair::new(100, 200),
        DelayPair::new(50, 250),
    );
    assert_eq!(dq.min_delay(), 50);
}

#[test]
fn delay_quad_max_delay() {
    let dq = DelayQuad::new(
        DelayPair::new(100, 200),
        DelayPair::new(50, 250),
    );
    assert_eq!(dq.max_delay(), 250);
}

#[test]
fn delay_quad_as_delay_pair() {
    let dq = DelayQuad::new(
        DelayPair::new(100, 200),
        DelayPair::new(50, 250),
    );
    let dp = dq.as_delay_pair();
    assert_eq!(dp.min_delay, 50);
    assert_eq!(dp.max_delay, 250);
}

#[test]
fn delay_quad_add() {
    let a = DelayQuad::new(
        DelayPair::new(100, 200),
        DelayPair::new(150, 250),
    );
    let b = DelayQuad::new(
        DelayPair::new(10, 20),
        DelayPair::new(15, 25),
    );
    let c = a + b;
    assert_eq!(c.rise.min_delay, 110);
    assert_eq!(c.rise.max_delay, 220);
    assert_eq!(c.fall.min_delay, 165);
    assert_eq!(c.fall.max_delay, 275);
}

#[test]
fn delay_quad_copy() {
    let a = DelayQuad::uniform(42);
    let b = a;
    assert_eq!(a, b);
}

#[test]
fn delay_quad_debug() {
    let dq = DelayQuad::uniform(100);
    let s = format!("{:?}", dq);
    assert!(s.contains("DelayQuad"));
    assert!(s.contains("rise"));
    assert!(s.contains("fall"));
}

// ============================================================================
// Tests from loc.rs
// ============================================================================

#[test]
fn loc_new_and_fields() {
    let loc = Loc::new(1, 2, 3);
    assert_eq!(loc.x, 1);
    assert_eq!(loc.y, 2);
    assert_eq!(loc.z, 3);
}

#[test]
fn loc_default_is_origin() {
    let loc = Loc::default();
    assert_eq!(loc.x, 0);
    assert_eq!(loc.y, 0);
    assert_eq!(loc.z, 0);
}

#[test]
fn loc_equality() {
    let a = Loc::new(1, 2, 3);
    let b = Loc::new(1, 2, 3);
    let c = Loc::new(1, 2, 4);
    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn loc_copy_semantics() {
    let a = Loc::new(5, 6, 7);
    let b = a;
    assert_eq!(a, b);
}

#[test]
fn loc_hashing() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(Loc::new(1, 2, 3));
    set.insert(Loc::new(4, 5, 6));
    set.insert(Loc::new(1, 2, 3));
    assert_eq!(set.len(), 2);
}

#[test]
fn manhattan_distance_same_point() {
    let a = Loc::new(5, 5, 0);
    assert_eq!(a.manhattan_distance(a), 0);
}

#[test]
fn manhattan_distance_horizontal() {
    let a = Loc::new(0, 0, 0);
    let b = Loc::new(5, 0, 0);
    assert_eq!(a.manhattan_distance(b), 5);
}

#[test]
fn manhattan_distance_vertical() {
    let a = Loc::new(0, 0, 0);
    let b = Loc::new(0, 3, 0);
    assert_eq!(a.manhattan_distance(b), 3);
}

#[test]
fn manhattan_distance_diagonal() {
    let a = Loc::new(0, 0, 0);
    let b = Loc::new(3, 4, 0);
    assert_eq!(a.manhattan_distance(b), 7);
}

#[test]
fn manhattan_distance_ignores_z() {
    let a = Loc::new(0, 0, 0);
    let b = Loc::new(0, 0, 100);
    assert_eq!(a.manhattan_distance(b), 0);
}

#[test]
fn manhattan_distance_negative_coords() {
    let a = Loc::new(-3, -4, 0);
    let b = Loc::new(3, 4, 0);
    assert_eq!(a.manhattan_distance(b), 14);
}

#[test]
fn loc_debug_format() {
    let loc = Loc::new(1, 2, 3);
    assert_eq!(format!("{:?}", loc), "Loc(1, 2, 3)");
}

#[test]
fn loc_display_format() {
    let loc = Loc::new(1, 2, 3);
    assert_eq!(format!("{}", loc), "(1, 2, 3)");
}

// ============================================================================
// Tests from enums.rs
// ============================================================================

// === PlaceStrength tests ===

#[test]
fn place_strength_default() {
    assert_eq!(PlaceStrength::default(), PlaceStrength::None);
}

#[test]
fn place_strength_is_locked() {
    assert!(!PlaceStrength::None.is_locked());
    assert!(!PlaceStrength::Weak.is_locked());
    assert!(!PlaceStrength::Strong.is_locked());
    assert!(!PlaceStrength::Placer.is_locked());
    assert!(PlaceStrength::Fixed.is_locked());
    assert!(PlaceStrength::Locked.is_locked());
    assert!(PlaceStrength::User.is_locked());
}

#[test]
fn place_strength_equality() {
    assert_eq!(PlaceStrength::None, PlaceStrength::None);
    assert_ne!(PlaceStrength::None, PlaceStrength::Weak);
}

#[test]
fn place_strength_copy() {
    let a = PlaceStrength::Fixed;
    let b = a;
    assert_eq!(a, b);
}

#[test]
fn place_strength_display() {
    assert_eq!(format!("{}", PlaceStrength::None), "NONE");
    assert_eq!(format!("{}", PlaceStrength::Weak), "WEAK");
    assert_eq!(format!("{}", PlaceStrength::Strong), "STRONG");
    assert_eq!(format!("{}", PlaceStrength::Placer), "PLACER");
    assert_eq!(format!("{}", PlaceStrength::Fixed), "FIXED");
    assert_eq!(format!("{}", PlaceStrength::Locked), "LOCKED");
    assert_eq!(format!("{}", PlaceStrength::User), "USER");
}

#[test]
fn place_strength_debug() {
    assert_eq!(format!("{:?}", PlaceStrength::Fixed), "Fixed");
}

#[test]
fn place_strength_hashing() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(PlaceStrength::None);
    set.insert(PlaceStrength::Fixed);
    set.insert(PlaceStrength::None);
    assert_eq!(set.len(), 2);
}

// === PortType tests ===

#[test]
fn port_type_default() {
    assert_eq!(PortType::default(), PortType::In);
}

#[test]
fn port_type_equality() {
    assert_eq!(PortType::In, PortType::In);
    assert_ne!(PortType::In, PortType::Out);
    assert_ne!(PortType::Out, PortType::InOut);
}

#[test]
fn port_type_display() {
    assert_eq!(format!("{}", PortType::In), "IN");
    assert_eq!(format!("{}", PortType::Out), "OUT");
    assert_eq!(format!("{}", PortType::InOut), "INOUT");
}

#[test]
fn port_type_debug() {
    assert_eq!(format!("{:?}", PortType::In), "In");
}

#[test]
fn port_type_copy() {
    let a = PortType::Out;
    let b = a;
    assert_eq!(a, b);
}

// === TimingPortClass tests ===

#[test]
fn timing_port_class_default() {
    assert_eq!(TimingPortClass::default(), TimingPortClass::Combinational);
}

#[test]
fn timing_port_class_is_register() {
    assert!(!TimingPortClass::Combinational.is_register());
    assert!(TimingPortClass::RegisterInput.is_register());
    assert!(TimingPortClass::RegisterOutput.is_register());
    assert!(!TimingPortClass::ClockInput.is_register());
    assert!(!TimingPortClass::GenClock.is_register());
    assert!(!TimingPortClass::Ignore.is_register());
}

#[test]
fn timing_port_class_is_clock() {
    assert!(!TimingPortClass::Combinational.is_clock());
    assert!(!TimingPortClass::RegisterInput.is_clock());
    assert!(!TimingPortClass::RegisterOutput.is_clock());
    assert!(TimingPortClass::ClockInput.is_clock());
    assert!(TimingPortClass::GenClock.is_clock());
    assert!(!TimingPortClass::Ignore.is_clock());
}

#[test]
fn timing_port_class_display() {
    assert_eq!(format!("{}", TimingPortClass::Combinational), "COMBINATIONAL");
    assert_eq!(format!("{}", TimingPortClass::RegisterInput), "REGISTER_INPUT");
    assert_eq!(format!("{}", TimingPortClass::RegisterOutput), "REGISTER_OUTPUT");
    assert_eq!(format!("{}", TimingPortClass::ClockInput), "CLOCK_INPUT");
    assert_eq!(format!("{}", TimingPortClass::GenClock), "GEN_CLOCK");
    assert_eq!(format!("{}", TimingPortClass::Ignore), "IGNORE");
}

#[test]
fn timing_port_class_equality() {
    assert_eq!(TimingPortClass::Combinational, TimingPortClass::Combinational);
    assert_ne!(TimingPortClass::Combinational, TimingPortClass::Ignore);
}

#[test]
fn timing_port_class_hashing() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(TimingPortClass::Combinational);
    set.insert(TimingPortClass::RegisterInput);
    set.insert(TimingPortClass::Combinational);
    assert_eq!(set.len(), 2);
}

// === ClockEdge tests ===

#[test]
fn clock_edge_default() {
    assert_eq!(ClockEdge::default(), ClockEdge::Rising);
}

#[test]
fn clock_edge_opposite() {
    assert_eq!(ClockEdge::Rising.opposite(), ClockEdge::Falling);
    assert_eq!(ClockEdge::Falling.opposite(), ClockEdge::Rising);
}

#[test]
fn clock_edge_double_opposite() {
    assert_eq!(ClockEdge::Rising.opposite().opposite(), ClockEdge::Rising);
    assert_eq!(ClockEdge::Falling.opposite().opposite(), ClockEdge::Falling);
}

#[test]
fn clock_edge_display() {
    assert_eq!(format!("{}", ClockEdge::Rising), "RISING");
    assert_eq!(format!("{}", ClockEdge::Falling), "FALLING");
}

#[test]
fn clock_edge_equality() {
    assert_eq!(ClockEdge::Rising, ClockEdge::Rising);
    assert_ne!(ClockEdge::Rising, ClockEdge::Falling);
}

#[test]
fn clock_edge_copy() {
    let a = ClockEdge::Rising;
    let b = a;
    assert_eq!(a, b);
}

#[test]
fn clock_edge_hashing() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(ClockEdge::Rising);
    set.insert(ClockEdge::Falling);
    set.insert(ClockEdge::Rising);
    assert_eq!(set.len(), 2);
}

// ============================================================================
// Tests from id_string.rs
// ============================================================================

#[test]
fn empty_string_is_zero() {
    assert_eq!(IdString::EMPTY.index(), 0);
    assert!(IdString::EMPTY.is_empty());
}

#[test]
fn id_string_default_is_empty() {
    let id = IdString::default();
    assert_eq!(id, IdString::EMPTY);
    assert!(id.is_empty());
}

#[test]
fn intern_returns_same_id_for_same_string() {
    let pool = IdStringPool::new();
    let a = pool.intern("hello");
    let b = pool.intern("hello");
    assert_eq!(a, b);
}

#[test]
fn intern_returns_different_ids_for_different_strings() {
    let pool = IdStringPool::new();
    let a = pool.intern("hello");
    let b = pool.intern("world");
    assert_ne!(a, b);
}

#[test]
fn intern_empty_string_returns_empty() {
    let pool = IdStringPool::new();
    let id = pool.intern("");
    assert_eq!(id, IdString::EMPTY);
}

#[test]
fn lookup_interned_string() {
    let pool = IdStringPool::new();
    let id = pool.intern("test");
    assert_eq!(pool.lookup(id), Some("test"));
}

#[test]
fn lookup_empty_id() {
    let pool = IdStringPool::new();
    assert_eq!(pool.lookup(IdString::EMPTY), Some(""));
}

#[test]
fn lookup_invalid_id() {
    let pool = IdStringPool::new();
    assert_eq!(pool.lookup(IdString(999)), None);
}

#[test]
fn pool_len() {
    let pool = IdStringPool::new();
    assert_eq!(pool.len(), 1); // empty string at index 0
    pool.intern("a");
    assert_eq!(pool.len(), 2);
    pool.intern("b");
    assert_eq!(pool.len(), 3);
    pool.intern("a"); // duplicate, no growth
    assert_eq!(pool.len(), 3);
}

#[test]
fn pool_is_empty() {
    let pool = IdStringPool::new();
    assert!(pool.is_empty());
    pool.intern("x");
    assert!(!pool.is_empty());
}

#[test]
fn ids_are_sequential() {
    let pool = IdStringPool::new();
    let a = pool.intern("first");
    let b = pool.intern("second");
    let c = pool.intern("third");
    assert_eq!(a.index(), 1);
    assert_eq!(b.index(), 2);
    assert_eq!(c.index(), 3);
}

#[test]
fn id_string_copy_semantics() {
    let id = IdString(42);
    let copy = id;
    assert_eq!(id, copy);
}

#[test]
fn id_string_hashing() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(IdString(1));
    set.insert(IdString(2));
    set.insert(IdString(1));
    assert_eq!(set.len(), 2);
}

#[test]
fn id_string_debug() {
    let id = IdString(42);
    assert_eq!(format!("{:?}", id), "IdString(42)");
}

#[test]
fn pool_debug() {
    let pool = IdStringPool::new();
    let debug = format!("{:?}", pool);
    assert!(debug.contains("IdStringPool"));
    assert!(debug.contains("count"));
}

#[test]
fn thread_safety() {
    use std::sync::Arc;
    use std::thread;

    let pool = Arc::new(IdStringPool::new());
    let mut handles = vec![];

    for i in 0..10 {
        let pool = Arc::clone(&pool);
        handles.push(thread::spawn(move || {
            let s = format!("string_{}", i);
            let id = pool.intern(&s);
            assert!(!id.is_empty());
            assert_eq!(pool.lookup(id), Some(s.as_str()));
            id
        }));
    }

    let ids: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    // All IDs should be distinct
    let mut unique = std::collections::HashSet::new();
    for id in &ids {
        unique.insert(*id);
    }
    assert_eq!(unique.len(), 10);
}

#[test]
fn concurrent_duplicate_inserts() {
    use std::sync::Arc;
    use std::thread;

    let pool = Arc::new(IdStringPool::new());
    let mut handles = vec![];

    // All threads intern the same string
    for _ in 0..10 {
        let pool = Arc::clone(&pool);
        handles.push(thread::spawn(move || pool.intern("shared")));
    }

    let ids: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    // All should get the same ID
    for id in &ids {
        assert_eq!(*id, ids[0]);
    }

    // Pool should only have 2 entries (empty + "shared")
    assert_eq!(pool.len(), 2);
}

#[test]
fn non_empty_id_is_not_empty() {
    let pool = IdStringPool::new();
    let id = pool.intern("notempty");
    assert!(!id.is_empty());
}

// ============================================================================
// Tests from property.rs
// ============================================================================

#[test]
fn string_property() {
    let p = Property::string("hello");
    assert!(p.is_string());
    assert!(!p.is_int());
    assert!(!p.is_bit_vector());
    assert_eq!(p.as_str(), "hello");
}

#[test]
fn int_property() {
    let p = Property::int(42);
    assert!(p.is_int());
    assert!(!p.is_string());
    assert!(!p.is_bit_vector());
    assert_eq!(p.as_int(), Some(42));
}

#[test]
fn bit_vector_property() {
    let p = Property::bit_vector("1010");
    assert!(p.is_bit_vector());
    assert!(!p.is_string());
    assert!(!p.is_int());
    assert_eq!(p.as_int(), Some(0b1010));
}

#[test]
fn bit_vector_with_unknown() {
    let p = Property::bit_vector("10x1");
    assert!(p.is_bit_vector());
    assert_eq!(p.as_int(), None); // cannot convert to int with 'x'
}

#[test]
fn bit_vector_empty() {
    let p = Property::bit_vector("");
    assert_eq!(p.as_int(), Some(0));
}

#[test]
fn string_as_int() {
    let p = Property::string("123");
    assert_eq!(p.as_int(), Some(123));
}

#[test]
fn string_not_int() {
    let p = Property::string("not_a_number");
    assert_eq!(p.as_int(), None);
}

#[test]
fn int_as_str() {
    let p = Property::int(42);
    assert_eq!(p.as_str(), "42");
}

#[test]
fn property_default_is_empty_string() {
    let p = Property::default();
    assert!(p.is_string());
    assert_eq!(p.as_str(), "");
}

#[test]
fn display_string() {
    let p = Property::string("test");
    assert_eq!(format!("{}", p), "test");
}

#[test]
fn display_int() {
    let p = Property::int(-5);
    assert_eq!(format!("{}", p), "-5");
}

#[test]
fn display_bit_vector() {
    let p = Property::bit_vector("1100");
    assert_eq!(format!("{}", p), "1100");
}

#[test]
fn from_str_ref() {
    let p: Property = "hello".into();
    assert_eq!(p, Property::String("hello".to_owned()));
}

#[test]
fn from_string() {
    let p: Property = String::from("hello").into();
    assert_eq!(p, Property::String("hello".to_owned()));
}

#[test]
fn from_i64() {
    let p: Property = 42i64.into();
    assert_eq!(p, Property::Int(42));
}

#[test]
fn from_i32() {
    let p: Property = 42i32.into();
    assert_eq!(p, Property::Int(42));
}

#[test]
fn from_bool_true() {
    let p: Property = true.into();
    assert_eq!(p, Property::Int(1));
}

#[test]
fn from_bool_false() {
    let p: Property = false.into();
    assert_eq!(p, Property::Int(0));
}

#[test]
fn property_equality() {
    assert_eq!(Property::int(1), Property::int(1));
    assert_ne!(Property::int(1), Property::int(2));
    assert_ne!(Property::int(1), Property::string("1"));
}

#[test]
fn property_clone() {
    let p = Property::string("test");
    let q = p.clone();
    assert_eq!(p, q);
}

#[test]
fn negative_int() {
    let p = Property::int(-100);
    assert_eq!(p.as_int(), Some(-100));
    assert_eq!(p.as_str(), "-100");
}

#[test]
fn bit_vector_all_zeros() {
    let p = Property::bit_vector("0000");
    assert_eq!(p.as_int(), Some(0));
}

#[test]
fn bit_vector_all_ones() {
    let p = Property::bit_vector("1111");
    assert_eq!(p.as_int(), Some(0b1111));
}

#[test]
fn property_hashing() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(Property::int(1));
    set.insert(Property::int(2));
    set.insert(Property::int(1));
    assert_eq!(set.len(), 2);
}
