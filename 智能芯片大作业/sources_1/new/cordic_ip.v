`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: tongji
// Engineer: zhanghaoran
// 
// Create Date: 2024/12/15 16:01:32
// Module Name: cordic_ip
// Description: 000 001 010 011 100 101
// angle range: -90~+90
//
//////////////////////////////////////////////////////////////////////////////////

module  cordic_ip
#(    
    parameter PIPELINE  = 16                        //pipe_max = 16
)
(
    input                               clk,
    input                               rst_n,
    
    input       signed  [31:0]          x_0,      //range:-90~+90
    input       signed  [31:0]          y_0,
    input       signed  [31:0]          z_0,    
    input               [2:0]           mode,     //2:1 for circle linear Hyperbola;0 for vector or revolve 
    input                               pre_valid,

    output  reg signed  [31:0]          x_n,     
    output  reg signed  [31:0]          y_n,    
    output  reg signed  [31:0]          z_n,
    output  reg                         post_valid

);//delay PIPELINE+3

wire signed [31:0]  angle_array[15:0];
                                          //degree*2^16
assign angle_array[0]  = 32'sd2949120;    //arctan(2^0)*2^16
assign angle_array[1]  = 32'sd1740992;    //arctan(2^1)*2^16
assign angle_array[2]  = 32'sd919872;     //14.0362*2^16
assign angle_array[3]  = 32'sd466944;     //7.1250*2^16
assign angle_array[4]  = 32'sd234368;     //3.5763*2^16
assign angle_array[5]  = 32'sd117312;     //1.7899*2^16
assign angle_array[6]  = 32'sd58688;      //0.8952*2^16
assign angle_array[7]  = 32'sd29312;      //0.4476*2^16
assign angle_array[8]  = 32'sd14656;      //0.2238*2^16
assign angle_array[9]  = 32'sd7360;       //0.1119*2^16
assign angle_array[10] = 32'sd3648;       //0.0560*2^16
assign angle_array[11] = 32'sd1856;	      //0.0280*2^16
assign angle_array[12] = 32'sd896;        //0.0140*2^16
assign angle_array[13] = 32'sd448;        //0.0070*2^16
assign angle_array[14] = 32'sd256;        //0.0035*2^16
assign angle_array[15] = 32'sd128;        //0.0018*2^16

wire signed [31:0]  alpha_array[15:0];

assign alpha_array[0]   =   32'sd35999;   //arctanh(2^-1)*65536
assign alpha_array[1]   =   32'sd16739;   //arctan(2^-2)*65536
assign alpha_array[2]   =   32'sd8235;
assign alpha_array[3]   =   32'sd4101;
assign alpha_array[4]   =   32'sd2049;
assign alpha_array[5]   =   32'sd1024;
assign alpha_array[6]   =   32'sd512;
assign alpha_array[7]   =   32'sd256;
assign alpha_array[8]   =   32'sd128;
assign alpha_array[9]   =   32'sd64;
assign alpha_array[10]  =   32'sd32;
assign alpha_array[11]  =   32'sd16;
assign alpha_array[12]  =   32'sd8;
assign alpha_array[13]  =   32'sd4;
assign alpha_array[14]  =   32'sd2;
assign alpha_array[15]  =   32'sd1;


localparam K        = 32'h09b74;	   //0.607253*2^16
localparam K_h      = 32'sd79137;      //1.207534*65536           

reg signed 	[31:0] 		            currentX[PIPELINE:0];
reg signed 	[31:0] 		            currentY[PIPELINE:0];
reg signed 	[31:0] 		            currentZ[PIPELINE:0];

reg         [2:0]   mode_map;


reg         [1:0]   site;
wire signed [15:0]  angle_t;

assign  angle_t = z_0 >>> 16;

always@(posedge clk or negedge rst_n)begin
   if(!rst_n)begin
       currentX[0]    <=  32'sd0;
       currentY[0]    <=  32'sd0;
       currentZ[0]    <=  32'sd0;
       mode_map       <=  3'b000;
       site           <=  2'd0;
   end
   else    begin
        currentX[0]    <=   x_0;
        currentY[0]    <=   y_0;
        mode_map       <=   mode;
 
        if((mode == 3'd0)&&(angle_t >= (-16'sd90))&&(angle_t <= 16'sd90))begin
            currentZ[0]    <=   z_0;
            site           <=  2'd1;
        end
        else  if((mode == 3'd0)&& (angle_t > 16'sd90) && (angle_t <= 16'sd180))begin
            currentZ[0]   <=  32'sd11796480 - z_0;                  //180*65536 = 11796480
            site          <=  2'd2;
        end
        else  if((mode == 3'd0)&& (angle_t < -16'sd90) && (angle_t >= -16'sd180))  begin
            currentZ[0]   <=  -32'sd11796480 - z_0;
            site          <=  2'd3;
        end
        else begin
            currentZ[0]    <=   z_0;
        end
   end
end
//DELAY 1

wire signed [31:0]                  next000X[PIPELINE-1:0];  
wire signed [31:0]                  next000Y[PIPELINE-1:0];  
wire signed [31:0]                  next000Z[PIPELINE-1:0];

wire signed [31:0]                  next001X[PIPELINE-1:0];  
wire signed [31:0]                  next001Y[PIPELINE-1:0];  
wire signed [31:0]                  next001Z[PIPELINE-1:0];

wire signed [31:0]                  next010X[PIPELINE-1:0];  
wire signed [31:0]                  next010Y[PIPELINE-1:0];  
wire signed [31:0]                  next010Z[PIPELINE-1:0];

wire signed [31:0]                  next011X[PIPELINE-1:0];  
wire signed [31:0]                  next011Y[PIPELINE-1:0];  
wire signed [31:0]                  next011Z[PIPELINE-1:0];

wire signed [31:0]                  next100X[PIPELINE-1:0];  
wire signed [31:0]                  next100Y[PIPELINE-1:0];  
wire signed [31:0]                  next100Z[PIPELINE-1:0];

wire signed [31:0]                  next101X[PIPELINE-1:0];  
wire signed [31:0]                  next101Y[PIPELINE-1:0];  
wire signed [31:0]                  next101Z[PIPELINE-1:0];

wire signed [31:0]                  temp100X[PIPELINE-1:0]; 
wire signed [31:0]                  temp100Y[PIPELINE-1:0];
wire signed [31:0]                  temp100Z[PIPELINE-1:0];

wire signed [31:0]                  temp101X[PIPELINE-1:0]; 
wire signed [31:0]                  temp101Y[PIPELINE-1:0];
wire signed [31:0]                  temp101Z[PIPELINE-1:0];

genvar i;
generate
    for(i = 1;i < PIPELINE+1;i = i + 1)begin : cal_xyz
        //sin/cos
        assign next000X[i-1] = (currentZ[i-1][31]) ? (currentX[i-1] + (currentY[i-1] >>> (i-1))) : (currentX[i-1] - (currentY[i-1] >>> (i-1)));
        assign next000Y[i-1] = (currentZ[i-1][31]) ? (currentY[i-1] - (currentX[i-1] >>> (i-1))) : (currentY[i-1] + (currentX[i-1] >>> (i-1)));
        assign next000Z[i-1] = (currentZ[i-1][31]) ? (currentZ[i-1] + angle_array[i-1]) : (currentZ[i-1] - angle_array[i-1]);

        //arctan
        assign next001X[i-1] = (currentY[i-1][31]) ? (currentX[i-1] - (currentY[i-1] >>> (i-1))) : (currentX[i-1] + (currentY[i-1] >>> (i-1)));
        assign next001Y[i-1] = (currentY[i-1][31]) ? (currentY[i-1] + (currentX[i-1] >>> (i-1))) : (currentY[i-1] - (currentX[i-1] >>> (i-1)));
        assign next001Z[i-1] = (currentY[i-1][31]) ? (currentZ[i-1] - angle_array[i-1]) : (currentZ[i-1] + angle_array[i-1]);

        //multiply
        assign next010X[i-1] =  currentX[i-1];
        assign next010Y[i-1] = (currentZ[i-1][31]) ? (currentY[i-1] - (currentX[i-1] >>> (i-1))) : (currentY[i-1] + (currentX[i-1] >>> (i-1)));
        assign next010Z[i-1] = (currentZ[i-1][31]) ? (currentZ[i-1] + ($signed(65536) >>> (i-1))) : (currentZ[i-1] - ($signed(65536) >>> (i-1)));

        //divided
        assign next011X[i-1] = currentX[i-1];
        assign next011Y[i-1] = ((currentY[i-1][31]^currentX[i-1][31]) == 1) ? (currentY[i-1] + (currentX[i-1] >>> (i-1))) : (currentY[i-1] - (currentX[i-1] >>> (i-1)));
        assign next011Z[i-1] = ((currentY[i-1][31]^currentX[i-1][31]) == 1) ? (currentZ[i-1] - ($signed(65536) >>> (i-1))) : (currentZ[i-1] + ($signed(65536) >>> (i-1)));

        //sinh/cosh
        assign next100X[i-1] = (currentZ[i-1][31]) ? (currentX[i-1] - (currentY[i-1] >>> i)) : (currentX[i-1] + (currentY[i-1] >>> i));
        assign next100Y[i-1] = (currentZ[i-1][31]) ? (currentY[i-1] - (currentX[i-1] >>> i)) : (currentY[i-1] + (currentX[i-1] >>> i));
        assign next100Z[i-1] = (currentZ[i-1][31]) ? (currentZ[i-1] + alpha_array[i-1]) : (currentZ[i-1] - alpha_array[i-1]);
        //i % 4 == 0 again
        assign temp100X[i-1] = (next100Z[i-1][31]) ? (next100X[i-1] - (next100Y[i-1] >>> i)) : (next100X[i-1] + (next100Y[i-1] >>> i));
        assign temp100Y[i-1] = (next100Z[i-1][31]) ? (next100Y[i-1] - (next100X[i-1] >>> i)) : (next100Y[i-1] + (next100X[i-1] >>> i));
        assign temp100Z[i-1] = (next100Z[i-1][31]) ? (next100Z[i-1] + alpha_array[i-1]) : (next100Z[i-1] - alpha_array[i-1]);

        //arctanh
        assign next101X[i-1] = (currentY[i-1][31]) ? (currentX[i-1] + (currentY[i-1] >>> i)) : (currentX[i-1] - (currentY[i-1] >>> i));
        assign next101Y[i-1] = (currentY[i-1][31]) ? (currentY[i-1] + (currentX[i-1] >>> i)) : (currentY[i-1] - (currentX[i-1] >>> i));
        assign next101Z[i-1] = (currentY[i-1][31]) ? (currentZ[i-1] - alpha_array[i-1]) : (currentZ[i-1] + alpha_array[i-1]);
        // i % 4 == 0 again
        assign temp101X[i-1] = (next101Y[i-1][31]) ? (next101X[i-1] + (next101Y[i-1] >>> i)) : (next101X[i-1] - (next101Y[i-1] >>> i));
        assign temp101Y[i-1] = (next101Y[i-1][31]) ? (next101Y[i-1] + (next101X[i-1] >>> i)) : (next101Y[i-1] - (next101X[i-1] >>> i));
        assign temp101Z[i-1] = (next101Y[i-1][31]) ? (next101Z[i-1] - alpha_array[i-1]) : (next101Z[i-1] + alpha_array[i-1]);

        always@(posedge clk or negedge rst_n)begin
            if(!rst_n)begin
                currentX[i]    <=  32'sd0;
                currentY[i]    <=  32'sd0;
                currentZ[i]    <=  32'sd0;
            end
            else begin
                case(mode_map)    //orinal XYZ
                    3'b000:begin //circle revolve
                        currentX[i] <= next000X[i-1];
                        currentY[i] <= next000Y[i-1];
                        currentZ[i] <= next000Z[i-1];
                    end
                    3'b001:begin //circle vector
                        currentX[i] <= next001X[i-1];
                        currentY[i] <= next001Y[i-1];
                        currentZ[i] <= next001Z[i-1];
                    end
                    3'b010:begin //linear revolve
                        currentX[i] <= next010X[i-1];
                        currentY[i] <= next010Y[i-1];
                        currentZ[i] <= next010Z[i-1];
                    end
                    3'b011:begin //linear vector
                        currentX[i] <= next011X[i-1];
                        currentY[i] <= next011Y[i-1];
                        currentZ[i] <= next011Z[i-1];
                    end
                    3'b100:begin //Hyperbola revolve
                        if (i % 4 == 0) begin
                            // i % 4 == 0 again
                            currentX[i] <= temp100X[i-1];
                            currentY[i] <= temp100Y[i-1];
                            currentZ[i] <= temp100Z[i-1];
                            end
                        else    begin
                            currentX[i] <= next100X[i-1];
                            currentY[i] <= next100Y[i-1];
                            currentZ[i] <= next100Z[i-1];
                        end
                    end
                    3'b101:begin //Hyperbola vector
                        if (i % 4 == 0) begin
                            // i % 4 == 0 again
                            currentX[i] <= temp101X[i-1];
                            currentY[i] <= temp101Y[i-1];
                            currentZ[i] <= temp101Z[i-1];
                            end 
                        else begin
                            currentX[i] <= next101X[i-1];
                            currentY[i] <= next101Y[i-1];
                            currentZ[i] <= next101Z[i-1];
                            end
                    end       
                endcase   
            end
        end 
    end
endgenerate
//DELAY PIPELINE+1

reg     [PIPELINE:0]      valid_r;
reg     [2*PIPELINE-1:0]    site_r;
always@(posedge clk )begin
    valid_r <=  {valid_r[PIPELINE-1:0],pre_valid};
    site_r  <=  {site_r[2*PIPELINE-3:0],site}; 
end

always@(posedge clk or negedge rst_n)begin
    if(!rst_n)
        post_valid  <=  0;
    else
        post_valid  <=  valid_r[PIPELINE];  
end

always@(posedge clk or negedge rst_n)begin
    if(!rst_n)begin
        x_n <=  0;
        y_n <=  0;
        z_n <=  0;
    end
    else    if(~valid_r[PIPELINE])begin
        x_n <=  0;
        y_n <=  0;
        z_n <=  0;
    end
    else    begin
        if(mode_map == 3'd0)begin
            case(site_r[2*PIPELINE-1:2*PIPELINE-2])
                2'd1:begin
                    x_n <=   currentX[PIPELINE];
                    y_n <=   currentY[PIPELINE];
                    z_n <=   currentZ[PIPELINE];
                end
                2'd2:begin
                    x_n <=   ~currentX[PIPELINE] + 1;
                    y_n <=   currentY[PIPELINE];
                    z_n <=   currentZ[PIPELINE];
                end
                2'd3:begin
                    x_n <=   ~currentX[PIPELINE] + 1;
                    y_n <=   currentY[PIPELINE];
                    z_n <=   currentZ[PIPELINE];
                end
                
            endcase
        end
        else begin
            x_n <=   currentX[PIPELINE];
            y_n <=   currentY[PIPELINE];
            z_n <=   currentZ[PIPELINE];
        end
           
    end
end


endmodule