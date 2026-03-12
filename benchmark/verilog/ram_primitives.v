// VTR RAM primitives - parameterized behavioral implementations
// These get synthesized to LUT-based storage by Yosys

module single_port_ram #(
    parameter ADDR_WIDTH = 7,
    parameter DATA_WIDTH = 32
) (
    input clk,
    input we,
    input [ADDR_WIDTH-1:0] addr,
    input [DATA_WIDTH-1:0] data,
    output reg [DATA_WIDTH-1:0] out
);
    reg [DATA_WIDTH-1:0] mem [0:(1<<ADDR_WIDTH)-1];
    always @(posedge clk) begin
        if (we)
            mem[addr] <= data;
        out <= mem[addr];
    end
endmodule

module dual_port_ram #(
    parameter ADDR_WIDTH = 7,
    parameter DATA_WIDTH = 32
) (
    input clk,
    input we1, we2,
    input [ADDR_WIDTH-1:0] addr1, addr2,
    input [DATA_WIDTH-1:0] data1, data2,
    output reg [DATA_WIDTH-1:0] out1, out2
);
    reg [DATA_WIDTH-1:0] mem [0:(1<<ADDR_WIDTH)-1];
    always @(posedge clk) begin
        if (we1)
            mem[addr1] <= data1;
        if (we2)
            mem[addr2] <= data2;
        out1 <= mem[addr1];
        out2 <= mem[addr2];
    end
endmodule
