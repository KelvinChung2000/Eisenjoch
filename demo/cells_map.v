// Techmap rules: map Yosys internal cells to example arch primitives.
// Pin names use bracket notation to match chipdb BEL pins: I[0], I[1], etc.

module \$lut (A, Y);
    parameter WIDTH = 0;
    parameter [63:0] LUT = 0;

    input [WIDTH-1:0] A;
    output Y;

    generate
        if (WIDTH == 1) begin
            LUT6 #(.INIT(LUT)) _TECHMAP_REPLACE_ (
                .\I[0] (A[0]), .\I[1] (1'b0), .\I[2] (1'b0), .\I[3] (1'b0), .\I[4] (1'b0), .\I[5] (1'b0), .F(Y)
            );
        end else if (WIDTH == 2) begin
            LUT6 #(.INIT(LUT)) _TECHMAP_REPLACE_ (
                .\I[0] (A[0]), .\I[1] (A[1]), .\I[2] (1'b0), .\I[3] (1'b0), .\I[4] (1'b0), .\I[5] (1'b0), .F(Y)
            );
        end else if (WIDTH == 3) begin
            LUT6 #(.INIT(LUT)) _TECHMAP_REPLACE_ (
                .\I[0] (A[0]), .\I[1] (A[1]), .\I[2] (A[2]), .\I[3] (1'b0), .\I[4] (1'b0), .\I[5] (1'b0), .F(Y)
            );
        end else if (WIDTH == 4) begin
            LUT6 #(.INIT(LUT)) _TECHMAP_REPLACE_ (
                .\I[0] (A[0]), .\I[1] (A[1]), .\I[2] (A[2]), .\I[3] (A[3]), .\I[4] (1'b0), .\I[5] (1'b0), .F(Y)
            );
        end else if (WIDTH == 5) begin
            LUT6 #(.INIT(LUT)) _TECHMAP_REPLACE_ (
                .\I[0] (A[0]), .\I[1] (A[1]), .\I[2] (A[2]), .\I[3] (A[3]), .\I[4] (A[4]), .\I[5] (1'b0), .F(Y)
            );
        end else if (WIDTH == 6) begin
            LUT6 #(.INIT(LUT)) _TECHMAP_REPLACE_ (
                .\I[0] (A[0]), .\I[1] (A[1]), .\I[2] (A[2]), .\I[3] (A[3]), .\I[4] (A[4]), .\I[5] (A[5]), .F(Y)
            );
        end else begin
            wire _TECHMAP_FAIL_ = 1;
        end
    endgenerate
endmodule

// Map clock-less $_FF_ (data register) to DFF for the simplified benchmark arch.
module \$_FF_ (input D, output Q);
    DFF _TECHMAP_REPLACE_ (.D(D), .CLK(1'b1), .Q(Q));
endmodule
