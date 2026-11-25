use std::fmt::Write as _;
use std::path::PathBuf;

struct Pod {
    upstream: &'static str,
    fields: &'static [&'static str],
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
        const PODS: &[Pod] = &[$(Pod {
            upstream: stringify!($upstream_pod),
            fields: &[$(upstream_field!($rust $($upstream)?)),*],
        }),*];
    };
}

include!("pods.rs");

fn probe_source() -> String {
    let mut sizes = String::new();
    let mut fields = String::new();
    for pod in PODS {
        let name = pod.upstream;
        writeln!(sizes, "    {{\"{name}\", sizeof({name})}},").unwrap();
        for field in pod.fields {
            writeln!(
                fields,
                "    {{\"{name}\", \"{field}\", offsetof({name}, {field}), sizeof({name}::{field})}},"
            )
            .unwrap();
        }
    }
    format!(
        "#include <cstddef>\n#include \"chipdb.h\"\nUSING_NEXTPNR_NAMESPACE\n\
         struct PodSize {{ const char *pod; size_t size; }};\n\
         struct FieldLayout {{ const char *pod; const char *field; size_t offset; size_t size; }};\n\
         extern \"C\" {{\n\
         extern const PodSize chipdb_layout_pod_sizes[] = {{\n{sizes}}};\n\
         extern const size_t chipdb_layout_pod_count = sizeof(chipdb_layout_pod_sizes) / sizeof(PodSize);\n\
         extern const FieldLayout chipdb_layout_fields[] = {{\n{fields}}};\n\
         extern const size_t chipdb_layout_field_count = sizeof(chipdb_layout_fields) / sizeof(FieldLayout);\n\
         }}\n"
    )
}

fn main() {
    let manifest = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let upstream = manifest.join("../../third_party/nextpnr");
    let header = upstream.join("himbaechel/chipdb.h");
    assert!(
        header.is_file(),
        "{} is missing. Check out the submodule with \
         `git submodule update --init --checkout -- third_party/nextpnr`.",
        header.display()
    );
    let probe = PathBuf::from(std::env::var_os("OUT_DIR").unwrap()).join("probe.cc");
    std::fs::write(&probe, probe_source()).unwrap();
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .define("NEXTPNR_NAMESPACE", "nextpnr_himbaechel")
        .define("ARCH_HIMBAECHEL", None)
        .flag("-Wno-invalid-offsetof")
        .include(upstream.join("himbaechel"))
        .include(upstream.join("common/kernel"))
        .file(&probe)
        .compile("chipdb_layout_probe");
    println!("cargo:rerun-if-changed=pods.rs");
    println!("cargo:rerun-if-changed={}", header.display());
    println!(
        "cargo:rerun-if-changed={}",
        upstream.join("common/kernel/relptr.h").display()
    );
}
