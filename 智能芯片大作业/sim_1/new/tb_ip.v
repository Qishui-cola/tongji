`timescale 1ns/1ns

module tb_ip();

parameter PERIOD = 10;
reg clk;
reg rst_n;
reg pre_valid;
reg signed [31:0] angle;
reg        [1:0] mode;
wire signed [31:0] 		result1;
wire signed [31:0] 		result2;
wire        post_valid;

initial begin
    
	clk = 0;
	rst_n <= 0;
	angle <= 'b0;
    pre_valid <= 1'b0;
    mode <= 2'd0;

	#100 rst_n <=1;
			
	#10 @(posedge clk) 
            pre_valid <= 1'b1;
            mode <= 2'b00;//sin cos
            angle <= 32'sd30 * $signed(65536);
    
	#10 @(posedge clk) 				    angle <= 32'sd60 * $signed(65536);
    #10 @(posedge clk) 				    angle <= 32'sd90 * $signed(65536);
    #10 @(posedge clk) 				    angle <= 32'sd120* $signed(65536);
    #10 @(posedge clk) 				    angle <= 32'sd150* $signed(65536);
    #10 @(posedge clk)pre_valid <= 1'b0;
    
    #200@(posedge clk)
        pre_valid <= 1'b1; 
        mode <= 2'b01;//arctan
        angle <= 32'sd1* $signed(65536); //45
    
    #10 @(posedge clk) 				    angle <= -32'sd1* $signed(65536);//-45
    #10 @(posedge clk) 				    angle <= 32'sd2* $signed(65536);//63.43
    #10 @(posedge clk) 				    angle <= -32'sd2* $signed(65536);//-63.43
    #10 @(posedge clk) 				    angle <= 32'sd3* $signed(65536);//71.565
	#10 @(posedge clk)pre_valid <= 1'b0;
	
	#200@(posedge clk)
        pre_valid <= 1'b1; 
        mode <= 2'b10;//cosh  sinh
        angle <= 32'sd1* $signed(65536); //1.543  1.1175
    
    #10 @(posedge clk) 				    angle <= -32'sd1* $signed(65536);//1.543 -1.1175
    #10 @(posedge clk) 				    angle <= 32'sd2* $signed(65536);//3.7621  3.6268
    #10 @(posedge clk) 				    angle <= -32'sd2* $signed(65536);//3.7621 3.6268
    #10 @(posedge clk) 				    angle <= 32'sd3* $signed(65536);//10.0676  10.0178
	#10 @(posedge clk)pre_valid <= 1'b0;
	
	#200@(posedge clk)
        pre_valid <= 1'b1; 
        mode <= 2'b11;//arctanh
        angle <= 32'sd1* $signed(32768); //0.549
    
    #10 @(posedge clk) 				    angle <= -32'sd1* $signed(32768);//-0.549
    #10 @(posedge clk) 				    angle <= 32'sd1* $signed(52428);//1.0986
    #10 @(posedge clk) 				    angle <= -32'sd1* $signed(52428);//-1.0986
    #10 @(posedge clk) 				    angle <= 32'sd0* $signed(65536);//0
	#10 @(posedge clk)pre_valid <= 1'b0;
	
	#100000 $stop;	
end 


always #(PERIOD/2) clk = ~clk;

cordic_ip
#(    
    .PIPELINE(16)                        //pipe数，�?大支�?16
)cordic_ip
(
    .clk            (clk  ),
    .rst_n          (rst_n),
                
    .angle          (angle),      
    .mode           (mode),
    .pre_valid      (pre_valid),
                
    .result1        (result1), 
    .result2        (result2),
    .post_valid     (post_valid)

);

endmodule

