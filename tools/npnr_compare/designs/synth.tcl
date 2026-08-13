# Synthesise a design to the example uarch's cell library (LUT4 + DFF).
#
# Mirrors nextpnr's own synth_generic.tcl so the netlist is exactly what the
# example uarch expects. IO buffers are not instantiated here -- the uarch's
# pack step derives them from top-level ports.
#
# Usage: yosys -c synth.tcl -- <src.v> <top> <W> <out.json>

set src   [lindex $argv 0]
set top   [lindex $argv 1]
set width [lindex $argv 2]
set out   [lindex $argv 3]
set LUT_K 4

set here [file dirname [file normalize $argv0]]

yosys read_verilog -lib $here/prims.v
yosys read_verilog $src
yosys chparam -set W $width $top
yosys hierarchy -check -top $top
yosys proc
yosys flatten
yosys tribuf -logic
yosys deminout
yosys synth -run coarse
yosys memory_map
yosys opt -full
yosys techmap -map +/techmap.v
yosys opt -fast
yosys dfflegalize -cell {$_DFF_P_} 0
yosys abc -lut $LUT_K -dress
yosys clean
yosys techmap -D LUT_K=$LUT_K -map $here/cells_map.v
# The example uarch trims nextpnr's own IOBs and assumes synthesis already
# inserted buffers, so every top-level port needs an explicit INBUF/OUTBUF.
yosys iopadmap -bits -inpad INBUF O:PAD -outpad OUTBUF I:PAD \
    -ignore INBUF PAD -ignore OUTBUF PAD
yosys clean
yosys hierarchy -check
yosys stat
yosys write_json $out
