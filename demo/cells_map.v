// Techmap rules: map Yosys internal cells to example arch primitives.
// Pin names use bracket notation to match chipdb BEL pins: I[0], I[1], etc.

module \$lut (A, Y);
    parameter WIDTH = 0;
    parameter LUT = 0;

    input [WIDTH-1:0] A;
    output Y;

    generate
        if (WIDTH == 1) begin
            LUT4 #(.INIT({8{LUT[1:0]}})) _TECHMAP_REPLACE_ (
                .\I[0] (A[0]), .\I[1] (1'b0), .\I[2] (1'b0), .\I[3] (1'b0), .F(Y)
            );
        end else if (WIDTH == 2) begin
            LUT4 #(.INIT({4{LUT[3:0]}})) _TECHMAP_REPLACE_ (
                .\I[0] (A[0]), .\I[1] (A[1]), .\I[2] (1'b0), .\I[3] (1'b0), .F(Y)
            );
        end else if (WIDTH == 3) begin
            LUT4 #(.INIT({2{LUT[7:0]}})) _TECHMAP_REPLACE_ (
                .\I[0] (A[0]), .\I[1] (A[1]), .\I[2] (A[2]), .\I[3] (1'b0), .F(Y)
            );
        end else if (WIDTH == 4) begin
            LUT4 #(.INIT(LUT[15:0])) _TECHMAP_REPLACE_ (
                .\I[0] (A[0]), .\I[1] (A[1]), .\I[2] (A[2]), .\I[3] (A[3]), .F(Y)
            );
        end else begin
            wire _TECHMAP_FAIL_ = 1;
        end
    endgenerate
endmodule

// Map clock-less $_FF_ (data register) to always-transparent DLATCH.
module \$_FF_ (input D, output Q);
    DLATCH _TECHMAP_REPLACE_ (.D(D), .G(1'b1), .Q(Q));
endmodule
