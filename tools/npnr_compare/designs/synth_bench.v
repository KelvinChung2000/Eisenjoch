// Synthetic benchmark design for the nextpnr/eisenjoch comparison harness.
//
// Deliberately boring and fully synthesisable to LUT4 + DFF: an LFSR feeding an
// accumulator, with every output bit driven by logic. Nothing is tied to a
// constant, because the example uarch rejects a top-level port driven by a
// constant driver -- which is what makes the shipped blinky.v unusable here.
//
// W scales the design: cell count grows roughly linearly, so the same source
// gives a range of sizes for the comparison without changing its structure.

module top #(
    parameter W = 16
) (
    input  wire         clk_pad,
    input  wire         rst,
    input  wire [W-1:0] din,
    output wire [W-1:0] dout
);

    // The example fabric feeds its clock ladder from a single GCLK_OUT pip that
    // exists only in the tile at X1Y0, so the clock buffer must be pinned there
    // or routing fails. Instantiated explicitly rather than left to iopadmap
    // because only this one port needs the constraint.
    wire clk;
    (* BEL = "X1Y0/IO0" *)
    INBUF clk_buf (
        .PAD(clk_pad),
        .O  (clk)
    );

    reg [W-1:0] lfsr;
    reg [W-1:0] acc;

    wire fb = lfsr[W-1] ^ lfsr[W-2] ^ lfsr[1] ^ lfsr[0];

    always @(posedge clk) begin
        if (rst) begin
            lfsr <= din;
            acc  <= din;
        end else begin
            lfsr <= {lfsr[W-2:0], fb};
            acc  <= acc + lfsr;
        end
    end

    assign dout = acc ^ {lfsr[0], lfsr[W-1:1]};

endmodule
