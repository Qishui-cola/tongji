//TCP server  汇总单机加速和不加速的文件
#define _WINSOCK_DEPRECATED_NO_WARNINGS

#pragma comment(lib,"ws2_32.lib")
#include <WinSock2.h>
#include <iostream>
#include <iomanip>  // 包含控制格式化的头文件
#include <immintrin.h> // SSE 指令集
#include <omp.h>       // 包含 OpenMP 头文件
#include "my_function.h"
using namespace std;

void shuffleArray(float arr[], int len);

#define MAX_THREADS 64
#define SUBDATANUM 2000*1000   //数据字块的长度
#define DATANUM (SUBDATANUM * MAX_THREADS)  //64M个数据

float rawFloatData[DATANUM];   //原始数据数组
float result[DATANUM];



int main() {
	LARGE_INTEGER frequency;
	LARGE_INTEGER start, end;
	double elapsedTime1;
	double elapsedTime2;
	QueryPerformanceFrequency(&frequency);

	//数据初始化
	std::cout << "Test Begin!\n";
	for (size_t i = 0; i < DATANUM; i++)
	{
		rawFloatData[i] = float(i + 1);
	}

	float floatSum1 = 0.0;
	{
		QueryPerformanceCounter(&start);//start  
		//计算求和
		floatSum1 = Sum(rawFloatData, DATANUM);

		QueryPerformanceCounter(&end);//end

		std::cout << std::fixed << std::setprecision(3);
		std::cout << "Sum result  is： " << floatSum1 << std::endl;
		elapsedTime1 = static_cast<double>(end.QuadPart - start.QuadPart) / frequency.QuadPart;
		std::cout << "Time Consume1:\t" << elapsedTime1 << endl;


		float floatSum2 = 0.0;
		QueryPerformanceCounter(&start);//start  

		//计算求和
		floatSum2 = sumSpeedUp(rawFloatData, DATANUM);

		QueryPerformanceCounter(&end);//end
		std::cout << std::fixed << std::setprecision(3);
		std::cout << "Speedup Sum result  is： " << floatSum2 << std::endl;
		elapsedTime2 = static_cast<double>(end.QuadPart - start.QuadPart) / frequency.QuadPart;
		std::cout << "Time Consume1:\t" << elapsedTime2 << endl;
		std::cout << "加速比:\t" << elapsedTime1 / elapsedTime2 << endl;

		float std_float = 1130722617.7269; //python高精度的结果直接赋给float，代表float能表示的最精确的值
		std::cout << "标准结果(python结果赋给float)： " << std_float << std::endl;

	}
	std::cout << endl;
	std::cout << endl;
	{
		QueryPerformanceCounter(&start);//start  
		//计算最大值
		float floatmax1 = Max(rawFloatData, DATANUM);

		QueryPerformanceCounter(&end);//end

		std::cout << std::fixed << std::setprecision(3);
		std::cout << "Max result  is： " << floatmax1 << std::endl;
		elapsedTime1 = static_cast<double>(end.QuadPart - start.QuadPart) / frequency.QuadPart;
		std::cout << "Time Consume1:\t" << elapsedTime1 << endl;


		float floatmax2 = 0.0;
		QueryPerformanceCounter(&start);//start  

		//计算最大
		floatmax2 = maxSpeedUp(rawFloatData, DATANUM);

		QueryPerformanceCounter(&end);//end
		std::cout << std::fixed << std::setprecision(3);
		std::cout << "Speedup Max result  is： " << floatmax2 << std::endl;
		elapsedTime2 = static_cast<double>(end.QuadPart - start.QuadPart) / frequency.QuadPart;
		std::cout << "Time Consume1:\t" << elapsedTime2 << endl;

		std::cout << "加速比:\t" << elapsedTime1 / elapsedTime2 << endl;
	}
	std::cout << endl;
	std::cout << endl;
	{
		//打乱数组
		//shuffleArray(rawFloatData, DATANUM);

		// 获取开始时间
		QueryPerformanceCounter(&start);

		// 排序并测量时间
		Sort(rawFloatData, DATANUM, result);

		// 获取结束时间
		QueryPerformanceCounter(&end);

		// 计算时间差（以秒为单位）
		elapsedTime1 = static_cast<double>(end.QuadPart - start.QuadPart) / frequency.QuadPart;
		std::cout << "Total Time Consumed: " << elapsedTime1 << " seconds." << std::endl;

		//检测排序是否正确
		int flag = 1;
		for (int i = 0; i < DATANUM - 1; i++)
			if (result[i] > result[i + 1])
			{
				std::cout << "排序测试未通过..." << std::endl;
				flag = false;
				break;
			}

		if (flag)
			std::cout << "排序测试通过..." << std::endl;


		// 获取开始时间
		QueryPerformanceCounter(&start);

		//加速排序
		sortSpeedUp(rawFloatData, DATANUM, result);

		// 获取结束时间
		QueryPerformanceCounter(&end);

		// 计算时间差（以秒为单位）
		elapsedTime2 = static_cast<double>(end.QuadPart - start.QuadPart) / frequency.QuadPart;
		std::cout << "Total Time Consumed: " << elapsedTime2 << " seconds." << std::endl;

		//检测排序是否正确
		flag = 1;
		for (int i = 0; i < DATANUM - 1; i++)
			if (result[i] > result[i + 1])
			{
				std::cout << "排序测试未通过..." << std::endl;
				flag = false;
				break;
			}
		if (flag)
			std::cout << "排序测试通过..." << std::endl;

		std::cout << "加速比:\t" << elapsedTime1 / elapsedTime2 << endl;
	}

	return 0;
}

