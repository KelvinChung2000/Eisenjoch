//! Diff utility for comparing named item collections.

use rustc_hash::FxHashMap;

/// Diff two slices of named items, returning (added, removed, changed) name lists.
pub fn diff_by_name<'a, T>(
    old_items: &'a [T],
    new_items: &'a [T],
    name_of: impl Fn(&'a T) -> &'a str,
    is_equal: impl Fn(&T, &T) -> bool,
) -> (Vec<String>, Vec<String>, Vec<String>) {
    let old_map: FxHashMap<&str, &T> = old_items.iter().map(|item| (name_of(item), item)).collect();
    let new_map: FxHashMap<&str, &T> = new_items.iter().map(|item| (name_of(item), item)).collect();

    let mut added = Vec::new();
    let mut changed = Vec::new();

    for (&name, new_item) in &new_map {
        match old_map.get(name) {
            Some(old_item) if !is_equal(old_item, new_item) => {
                changed.push(name.to_string());
            }
            None => added.push(name.to_string()),
            _ => {}
        }
    }

    let removed: Vec<String> = old_map
        .keys()
        .filter(|name| !new_map.contains_key(*name))
        .map(|name| name.to_string())
        .collect();

    (added, removed, changed)
}
