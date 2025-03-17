`timescale 1ns / 1ps

module cordic_top (
    input                 clk,
    input                 rst_n,
    input       [3:0]     mode,        // 功能选择信号
    input  signed [31:0]  angle,    // 通用输入 A
    input                 pre_valid,   // 输入有效信号
 
    output reg signed [31:0] result,   // 输出结果 1
    output reg               post_valid   // 输出有效信号
);

localparam K        = 32'h09b74;	   //0.607253*2^16
localparam K_h      = 32'sd79137;      //1.207534*65536   

// sin/cos 
wire signed [31:0] sin_out, cos_out;
wire               sin_cos_valid;
cordic_ip
#(    
    .PIPELINE(16)                        
)sin_cos_cordic (
        .clk(clk),
        .rst_n(rst_n),
        .x_0(K),      
        .y_0(0),      
        .z_0(angle), 
        .mode(3'b000),     
        .pre_valid(pre_valid),
        .x_n(cos_out),      
        .y_n(sin_out),             
        .z_n(),             
        .post_valid(sin_cos_valid)
    );

// arctan
wire signed [31:0] arctan_out;
wire               arctan_valid;
cordic_ip
#(    
    .PIPELINE(16)                        
)arctan_cordic (
        .clk(clk),
        .rst_n(rst_n),
        .x_0(32'sd1 * $signed(65536)),      
        .y_0(angle),      
        .z_0(0), 
        .mode(3'b001),    
        .pre_valid(pre_valid),
        .x_n(),      
        .y_n(),             
        .z_n(arctan_out),             
        .post_valid(arctan_valid)
    );

// sinh/cosh
wire signed [31:0] sinh_out, cosh_out;
wire               sinh_cosh_valid;
cordic_ip
#(    
    .PIPELINE(16)                        
)sinh_cosh_cordic (
        .clk(clk),
        .rst_n(rst_n),
        .x_0(K_h),      // 除数
        .y_0(0),      // 被除数
        .z_0(angle), // 初始 z 值
        .mode(3'b100),      // 线性模式计算除法
        .pre_valid(pre_valid),
        .x_n(cosh_out),      // 输出 tan 值
        .y_n(sinh_out),             // 无需 y 输出
        .z_n(),             // 无需 z 输出
        .post_valid(sinh_cosh_valid)
    );
    
// arctanh
wire signed [31:0] arctanh_out;
wire               arctanh_valid;
cordic_ip
#(    
    .PIPELINE(16)                        
)arctanh_cordic (
        .clk(clk),
        .rst_n(rst_n),
        .x_0(32'sd1 * $signed(65536)),      // 除数
        .y_0(angle),      // 被除数
        .z_0(0), // 初始 z 值
        .mode(3'b101),      // 线性模式计算除法
        .pre_valid(pre_valid),
        .x_n(),      // 输出 tan 值
        .y_n(),             // 无需 y 输出
        .z_n(arctanh_out),             // 无需 z 输出
        .post_valid(arctanh_valid)
    );
    
// arcsin/arccos
wire signed [31:0] arcsin_out, arccos_out;
wire               arcsin_arccos_valid;
cordic_arcsin_arccos
#( 
    .PIPELINE(16)                        
)cordic_arcsin_arccos (
        .clk(clk),
        .rst_n(rst_n),
        .iData(angle),
        .pre_vaild(pre_valid),
        .arcsin(arcsin_out),      
        .arccos(arccos_out),             
        .post_vaild(arcsin_arccos_valid)
    );
    
// ln
wire signed [31:0] In_out;
wire               In_valid;
cordic_In
#( 
    .PIPELINE(16)                        
)In_cordic (
        .clk(clk),
        .rst_n(rst_n),
        .iData(angle),
        .pre_vaild(pre_valid),
        .In(In_out),                   
        .post_vaild(In_valid)
    );
    
// e
wire signed [31:0] e_out;
wire               e_valid;
cordic_exp_plus
#( 
    .WII(16),  
    .WOI(16),
    .WOF(16),    
    .PIPELINE(16)                  
)exp_cordic (
        .clk(clk),
        .rst_n(rst_n),
        .iData(angle),
        .pre_vaild(pre_valid),
        .exp(e_out),                   
        .post_vaild(e_valid)
    );
    

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        result <= 32'sd0;
        post_valid <= 1'b0;
    end else begin
        case (mode)
            4'b0000: begin // sin
                result <= sin_out;
                post_valid <= sin_cos_valid;
            end
            4'b0001: begin // cos
                result <= cos_out;
                post_valid <= sin_cos_valid;
            end
            4'b0010: begin //arctan
                result <= arctan_out;
                post_valid <= arctan_valid;
            end
            4'b0011: begin //sinh
                result <= sinh_out;
                post_valid <= sinh_cosh_valid;
            end
            4'b0100: begin //cosh
                result <= cosh_out;
                post_valid <= sinh_cosh_valid;
            end
            4'b0101: begin //atctanh
                result <= arctanh_out;
                post_valid <= arctanh_valid;
            end
            4'b0110: begin //arcsin
                result <= arcsin_out;
                post_valid <= arcsin_arccos_valid;
            end
            4'b0111: begin //arccos
                result <= arccos_out;
                post_valid <= arcsin_arccos_valid;
            end
            4'b1000: begin //ln
                result <= In_out;
                post_valid <= In_valid;
            end
            4'b1001: begin //exp
                result <= e_out;
                post_valid <= e_valid;
            end
            default: begin
                result <= 32'sd0;
                post_valid <= 1'b0;
            end
        endcase
    end
end




endmodule
