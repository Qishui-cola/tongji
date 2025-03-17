`timescale 1ns/1ns

module tb_cordic_top();

parameter PERIOD = 10;
reg clk;
reg rst_n;
reg [3:0]   mode;
reg signed [31:0] angle; 
reg pre_valid;

wire signed [31:0] 		result;
wire        post_valid;

initial begin
    
	clk = 0;
	rst_n <= 0;
	angle <= 'b0;
    pre_valid <= 1'b0;

	#100 rst_n <=1;
			
	#10 @(posedge clk) 
        pre_valid <= 1'b1;
        mode <= 4'b000; //sin
        angle <= 32'sd30 * $signed(65536);
    
	#10 @(posedge clk) 				    
        angle <= 32'sd60 * $signed(65536);
        
    #10 @(posedge clk) 				    
        pre_valid <= 1'b0;

    
	#200 @(posedge clk) 
        pre_valid <= 1'b1;
        mode <= 4'b001; //cos
        angle <= 32'sd30 * $signed(65536);
    
	#10 @(posedge clk) 				    
        angle <= 32'sd60 * $signed(65536);
    #10 @(posedge clk) 				    
        pre_valid <= 1'b0;
        
    #200 @(posedge clk) 
        pre_valid <= 1'b1;
        mode <= 4'b0010; //atctan
        angle <= 32'sd113508; //1.732
   
	#10 @(posedge clk) 				    
        angle <= 32'sd1 * $signed(65536);
    #10 @(posedge clk) 				    
        pre_valid <= 1'b0;  
        
    #200 @(posedge clk) 
        pre_valid <= 1'b1;
        mode <= 4'b0011; //sinh  
        angle <= 32'sd1 * $signed(65536);
    #10 @(posedge clk) 				    
        pre_valid <= 1'b0;
        
    #200 @(posedge clk) 
        pre_valid <= 1'b1;
        mode <= 4'b0100; //cosh
        angle <= 32'sd1 * $signed(65536);
    #10 @(posedge clk) 				    
        pre_valid <= 1'b0;
    
    #200 @(posedge clk) 
        pre_valid <= 1'b1;
        mode <= 4'b0101; //arctanh
        angle <= 32'sd1 * $signed(6553);
    #10 @(posedge clk) 				    
        pre_valid <= 1'b0;
    
    #200 @(posedge clk) 
        pre_valid <= 1'b1;
        mode <= 4'b0110; //arcsin
        angle <= 32'sd1 * $signed(32768);
    #10 @(posedge clk) 				    
        pre_valid <= 1'b0;
    
    #200 @(posedge clk) 
        pre_valid <= 1'b1;
        mode <= 4'b0111; //arccos
        angle <= 32'sd1 * $signed(32768);
    #10 @(posedge clk) 				    
        pre_valid <= 1'b0;
        
    #200 @(posedge clk) 
        pre_valid <= 1'b1;
        mode <= 4'b1000; //ln
        angle <= 32'sd2 * $signed(65536);
    #10 @(posedge clk) 				    
        pre_valid <= 1'b0;
        
    #200 @(posedge clk) 
        pre_valid <= 1'b1;
        mode <= 4'b1001; //exp
        angle <= 32'sd1 * $signed(65536);
    #10 @(posedge clk) 				    
        pre_valid <= 1'b0;
	#100000 $stop;	
end 


always #(PERIOD/2) clk = ~clk;

cordic_top cordic_top
(
    .clk        (clk  ),
    .rst_n      (rst_n),
    .mode      (mode), 
    .angle      (angle),             
    .pre_valid  (pre_valid),
                
    .result   (result), 
    .post_valid (post_valid)

);



endmodule

