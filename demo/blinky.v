// Simple blinky demo for nextpnr-rust example architecture.
// Toggles an LED output using a counter driven by a clock.

module top (
    input  wire clk,
    output wire led
);

    reg [3:0] counter;

    always @(posedge clk)
        counter <= counter + 1'b1;

    assign led = counter[3];

endmodule
