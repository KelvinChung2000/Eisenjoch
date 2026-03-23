// Simulation models for the example architecture primitives.
// These define the cell interface that Yosys maps into.
// Pin names use bracket notation to match the chipdb BEL pin names.

(* blackbox *)
module LUT6 (
    input  \I[0] , \I[1] , \I[2] , \I[3] , \I[4] , \I[5] ,
    output F
);
    parameter [63:0] INIT = 64'h0000000000000000;

    wire [5:0] s = {\I[5] , \I[4] , \I[3] , \I[2] , \I[1] , \I[0] };
    assign F = INIT[s];
endmodule

(* blackbox *)
module LUT4 (
    input  \I[0] , \I[1] , \I[2] , \I[3] ,
    output F
);
    parameter [15:0] INIT = 16'h0000;

    wire [3:0] s = {\I[3] , \I[2] , \I[1] , \I[0] };
    assign F = INIT[s];
endmodule

(* blackbox *)
module DFF (
    input  D, CLK,
    output reg Q
);
    always @(posedge CLK)
        Q <= D;
endmodule

(* blackbox *)
module IOB (
    input  I, T,
    output O,
    inout  PAD
);
    assign PAD = T ? 1'bz : I;
    assign O = PAD;
endmodule
