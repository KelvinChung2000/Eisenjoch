fn main() {
    for line in nextpnr::placer::opt_trans::debug_lbfgs_smoke_trace() {
        println!("{line}");
    }
}
