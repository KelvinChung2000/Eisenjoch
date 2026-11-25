#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};
    use std::ffi::{c_char, CStr};
    use std::fmt::Debug;
    use std::mem::{offset_of, size_of};

    use nextpnr_db::chipdb::*;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct FieldSpan {
        offset: usize,
        size: usize,
    }

    type FieldKey = (String, String);

    #[derive(Default)]
    struct Layout {
        pod_sizes: BTreeMap<String, usize>,
        fields: BTreeMap<FieldKey, FieldSpan>,
    }

    #[repr(C)]
    struct PodSize {
        pod: *const c_char,
        size: usize,
    }

    #[repr(C)]
    struct FieldLayout {
        pod: *const c_char,
        field: *const c_char,
        offset: usize,
        size: usize,
    }

    extern "C" {
        #[link_name = "chipdb_layout_pod_sizes"]
        static POD_SIZES: [PodSize; 0];
        #[link_name = "chipdb_layout_pod_count"]
        static POD_COUNT: usize;
        #[link_name = "chipdb_layout_fields"]
        static FIELDS: [FieldLayout; 0];
        #[link_name = "chipdb_layout_field_count"]
        static FIELD_COUNT: usize;
    }

    fn owned(name: *const c_char) -> String {
        unsafe { CStr::from_ptr(name) }.to_str().unwrap().to_owned()
    }

    fn upstream_layout() -> Layout {
        let pod_sizes = unsafe { std::slice::from_raw_parts(POD_SIZES.as_ptr(), POD_COUNT) };
        let fields = unsafe { std::slice::from_raw_parts(FIELDS.as_ptr(), FIELD_COUNT) };
        Layout {
            pod_sizes: pod_sizes
                .iter()
                .map(|row| (owned(row.pod), row.size))
                .collect(),
            fields: fields
                .iter()
                .map(|row| {
                    let span = FieldSpan {
                        offset: row.offset,
                        size: row.size,
                    };
                    ((owned(row.pod), owned(row.field)), span)
                })
                .collect(),
        }
    }

    fn field_size<Pod, Field>(_: fn(Pod) -> Field) -> usize {
        size_of::<Field>()
    }

    macro_rules! upstream_field {
        ($rust:ident) => {
            stringify!($rust)
        };
        ($rust:ident $upstream:ident) => {
            stringify!($upstream)
        };
    }

    macro_rules! pods {
        ($($rust_pod:ident => $upstream_pod:ident { $($rust:ident $(=> $upstream:ident)?),* $(,)? }),* $(,)?) => {
            fn rust_layout() -> Layout {
                let mut layout = Layout::default();
                $(
                    let pod = stringify!($upstream_pod).to_owned();
                    layout.pod_sizes.insert(pod.clone(), size_of::<$rust_pod>());
                    $(
                        layout.fields.insert(
                            (pod.clone(), upstream_field!($rust $($upstream)?).to_owned()),
                            FieldSpan {
                                offset: offset_of!($rust_pod, $rust),
                                size: field_size(|p: $rust_pod| p.$rust),
                            },
                        );
                    )*
                )*
                layout
            }
        };
    }

    include!("../pods.rs");

    fn mismatches<K: Ord + Clone + Debug, V: PartialEq + Copy + Debug>(
        rust: &BTreeMap<K, V>,
        upstream: &BTreeMap<K, V>,
    ) -> Vec<(K, Option<V>, Option<V>)> {
        let keys: BTreeSet<&K> = rust.keys().chain(upstream.keys()).collect();
        keys.into_iter()
            .map(|key| {
                (
                    key.clone(),
                    rust.get(key).copied(),
                    upstream.get(key).copied(),
                )
            })
            .filter(|(_, rust, upstream)| rust != upstream)
            .collect()
    }

    #[test]
    fn pod_sizes_match_upstream() {
        let diff = mismatches(&rust_layout().pod_sizes, &upstream_layout().pod_sizes);
        assert!(diff.is_empty(), "(pod, nextpnr-db, upstream): {diff:#?}");
    }

    #[test]
    fn table_lists_every_byte_of_each_pod() {
        let layout = upstream_layout();
        let mut listed: BTreeMap<String, usize> = BTreeMap::new();
        for ((pod, _), span) in &layout.fields {
            *listed.entry(pod.clone()).or_default() += span.size;
        }
        let diff = mismatches(&listed, &layout.pod_sizes);
        assert!(
            diff.is_empty(),
            "(pod, listed field bytes, sizeof): {diff:#?}"
        );
    }

    #[test]
    fn field_spans_match_upstream() {
        let diff = mismatches(&rust_layout().fields, &upstream_layout().fields);
        assert!(
            diff.is_empty(),
            "((pod, field), nextpnr-db, upstream): {diff:#?}"
        );
    }
}
