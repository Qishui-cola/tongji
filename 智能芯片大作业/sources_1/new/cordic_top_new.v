`timescale 1ns / 1ps

module cordic_top_new (
    input                 clk,
    input                 rst_n,
    input       [3:0]     mode,        // 功能选择信号
    input  signed [31:0]  angle,    // 通用输入 A
    input              pre_valid,   // 输入有效信号
 
    output reg signed [31:0] result,   // 输出结果 1
    output reg               post_valid   // 输出有效信号
);

localparam K        = 32'h09b74;	   //0.607253*2^16
localparam K_h      = 32'sd79137;      //1.207534*65536   

reg signed  [31:0] x_in;
reg signed  [31:0] y_in;
reg signed  [31:0] z_in;
reg         [2:0]  mode_in;
reg                in_valid;

wire signed [31:0] x_out, y_out,z_out;
wire               result_valid;

cordic_ip
#(    
    .PIPELINE(16)                        
)cordic_ip (
        .clk(clk),
        .rst_n(rst_n),
        .x_0(x_in),      
        .y_0(y_in),      
        .z_0(z_in), 
        .mode(mode_in),     
        .pre_valid(in_valid),
        .x_n(x_out),      
        .y_n(y_out),             
        .z_n(z_out),             
        .post_valid(result_valid)
    );

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        in_valid <= 0;
        result <= 32'sd0;
        post_valid <= 1'b0;
        mode_in <= 3'b000;
        x_in <= 32'sd0;
        y_in <= 32'sd0;
        z_in <= 32'sd0;
    end else begin
        case (mode)
            4'b0000: begin // 0 cos
                in_valid <= pre_valid;
                mode_in <= 3'b000;
                x_in <= K;
                y_in <= 0;
                z_in <= angle;
                result <= x_out;
                post_valid <= result_valid;
            end
            4'b0001: begin // 1 sin
                in_valid <= pre_valid;
                mode_in <= 3'b000;
                x_in <= K;
                y_in <= 0;
                z_in <= angle;
                result <= y_out;
                post_valid <= result_valid;
            end
            4'b0010: begin // 2 arctan
                in_valid <= pre_valid;
                mode_in <= 3'b001;
                x_in <= 32'sd1 * $signed(65536);
                y_in <= angle;
                z_in <= 0;
                result <= z_out;
                post_valid <= result_valid;
            end
            4'b0011: begin // 3 cosh
                in_valid <= pre_valid;
                mode_in <= 3'b100;
                x_in <= K_h;
                y_in <= 0;
                z_in <= angle;
                result <= x_out;
                post_valid <= result_valid;
            end
            4'b0100: begin // 4 sinh
                in_valid <= pre_valid;
                mode_in <= 3'b100;
                x_in <= K_h;
                y_in <= 0;
                z_in <= angle;
                result <= y_out;
                post_valid <= result_valid;
            end
            4'b0101: begin // 5 atctanh
                in_valid <= pre_valid;
                mode_in <= 3'b101;
                x_in <= 32'sd1 * $signed(65536);
                y_in <= angle;
                z_in <= 0;
                result <= z_out;
                post_valid <= result_valid;
            end
            4'b0110: begin // 6 genghao
                in_valid <= pre_valid;
                mode_in <= 3'b101;
                x_in <= angle + 32'sd1 * $signed(65536);
                y_in <= angle - 32'sd1 * $signed(65536);
                z_in <= 0;
                result <= x_out;
                post_valid <= result_valid;
            end
            4'b0111: begin //  7 multiple
                in_valid <= pre_valid;
                mode_in <= 3'b010;
                x_in <= angle;
                y_in <= 0;
                z_in <= angle;
                result <= y_out;
                post_valid <= result_valid;
            end
            4'b1000: begin //  9 multiple
                in_valid <= pre_valid;
                mode_in <= 3'b011;
                x_in <= angle + 32'sd1 * $signed(65536);
                y_in <= angle - 32'sd1 * $signed(65536);
                z_in <= 0;
                result <= z_out;
                post_valid <= result_valid;
            end
            default: begin
                in_valid <= 0;
                result <= 32'sd0;
                post_valid <= 1'b0;
                mode_in <= 3'b000;
                x_in <= 32'sd0;
                y_in <= 32'sd0;
                z_in <= 32'sd0;
            end
        endcase
    end
end




endmodule
