mod common;

use std::sync::Arc;
use std::time::{Duration, Instant};

use nextpnr::context::Context;
use nextpnr::live::{Cancelled, Live, LiveProgress};
use nextpnr::placer::heap::{place_heap, PlacerHeapCfg};
use nextpnr::placer::PlacerError;
use nextpnr::routed_json::routed_json;
use nextpnr::router::common::collect_routable_nets;
use nextpnr::router::router2::Router2Cfg;
use nextpnr::router::{Router, Router2, RouterError};
use serde_json::{Map, Value};

fn watched(mut ctx: Context) -> (Context, Arc<Live>) {
    let live = Live::new();
    ctx.attach_live(Arc::clone(&live));
    (ctx, live)
}

fn route(ctx: &mut Context) -> Result<(), RouterError> {
    let nets = collect_routable_nets(ctx);
    Router2.route_nets(ctx, &Router2Cfg::default(), &nets)
}

fn routed_attributes(ctx: &Context, kind: &str, attribute: &str) -> Map<String, Value> {
    let json = routed_json(ctx).expect("routed JSON");
    let module = json["modules"]
        .as_object()
        .unwrap()
        .values()
        .next()
        .unwrap();
    module[kind]
        .as_object()
        .unwrap()
        .iter()
        .filter_map(|(name, item)| {
            let value = item["attributes"][attribute].as_str()?.trim_end();
            (!value.is_empty()).then(|| (name.clone(), Value::from(value)))
        })
        .collect()
}

fn names(map: &std::collections::BTreeMap<String, String>) -> Map<String, Value> {
    map.iter()
        .map(|(name, value)| (name.clone(), Value::from(value.as_str())))
        .collect()
}

#[test]
fn a_request_captures_router2_at_its_next_offer() {
    let (mut ctx, live) = watched(common::placed_npnr_baseline());
    assert!(live.latest().is_none());
    ctx.publish_live(LiveProgress::bound("placed", None));
    let placed = live.latest().expect("a published frame");
    assert_eq!(placed.frame.version, 1);
    assert_eq!(placed.frame.phase, "placed");
    assert!(placed.frame.nets.is_empty());
    assert_eq!(placed.netlist.as_ref(), &routed_json(&ctx).unwrap());
    let cells = routed_attributes(&ctx, "cells", "NEXTPNR_BEL");
    assert_eq!(names(&placed.frame.cells), cells);

    live.request();
    route(&mut ctx).expect("negotiation converges");
    let negotiating = live.latest().unwrap();
    assert_eq!(negotiating.frame.version, 2);
    assert_eq!(negotiating.frame.phase, "route");
    assert_eq!(negotiating.frame.iteration, Some(0));
    assert!(negotiating.frame.overused.is_some());
    let routing = routed_attributes(&ctx, "netnames", "ROUTING");
    assert!(!negotiating.frame.nets.is_empty());
    assert!(negotiating
        .frame
        .nets
        .keys()
        .all(|net| routing.contains_key(net)));
    assert!(Arc::ptr_eq(&negotiating.netlist, &placed.netlist));

    ctx.publish_live(LiveProgress::bound("routed", None));
    let routed = live.latest().unwrap();
    assert_eq!(routed.frame.version, 3);
    assert_eq!(names(&routed.frame.cells), cells);
    assert_eq!(names(&routed.frame.nets), routing);
}

#[test]
fn an_unwatched_flow_captures_nothing() {
    let (mut ctx, live) = watched(common::placed_npnr_baseline());
    ctx.publish_live(LiveProgress::bound("placed", None));
    route(&mut ctx).expect("negotiation converges");
    assert_eq!(live.latest().unwrap().frame.phase, "placed");
}

#[test]
fn a_request_captures_each_heap_legalisation() {
    let (mut ctx, live) = watched(common::make_example_context_with_cells(8));
    live.request();
    place_heap(&mut ctx, &PlacerHeapCfg::default()).expect("HeAP places");
    let frame = live.latest().unwrap().frame;
    assert_eq!((frame.phase.as_str(), frame.iteration), ("place", Some(0)));
    assert_eq!(frame.cells.len(), 8);
}

#[test]
fn cancelling_stops_the_placer_and_the_router() {
    let (mut ctx, live) = watched(common::make_example_context_with_cells(8));
    live.cancel();
    assert!(matches!(
        place_heap(&mut ctx, &PlacerHeapCfg::default()),
        Err(PlacerError::Cancelled(Cancelled))
    ));
    let (mut ctx, live) = watched(common::placed_npnr_baseline());
    live.cancel();
    assert!(matches!(
        route(&mut ctx),
        Err(RouterError::Cancelled(Cancelled))
    ));
}

#[test]
fn a_viewer_thread_sees_versions_rise_while_the_router_runs() {
    let (mut ctx, live) = watched(common::placed_npnr_baseline());
    let viewer = {
        let live = Arc::clone(&live);
        std::thread::spawn(move || {
            let started = Instant::now();
            let mut versions = Vec::new();
            loop {
                live.request();
                if let Some(latest) = live.latest() {
                    if versions.last() != Some(&latest.frame.version) {
                        versions.push(latest.frame.version);
                    }
                    if latest.frame.phase == "routed" {
                        return versions;
                    }
                }
                assert!(started.elapsed() < Duration::from_secs(60), "never routed");
                std::thread::sleep(Duration::from_millis(1));
            }
        })
    };
    route(&mut ctx).expect("negotiation converges");
    ctx.publish_live(LiveProgress::bound("routed", None));
    let versions = viewer.join().unwrap();
    assert!(
        versions.windows(2).all(|pair| pair[0] < pair[1]),
        "{versions:?}"
    );
    assert_eq!(
        versions.last(),
        live.latest().map(|latest| latest.frame.version).as_ref()
    );
}
