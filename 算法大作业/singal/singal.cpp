//TCP server  ���ܵ������ٺͲ����ٵ��ļ�
#define _WINSOCK_DEPRECATED_NO_WARNINGS

#pragma comment(lib,"ws2_32.lib")
#include <WinSock2.h>
#include <iostream>
#include <iomanip>  // �������Ƹ�ʽ����ͷ�ļ�
#include <immintrin.h> // SSE ָ�
#include <omp.h>       // ���� OpenMP ͷ�ļ�
#include "my_function.h"
using namespace std;

void shuffleArray(float arr[], int len);

#define MAX_THREADS 64
#define SUBDATANUM 2000*1000   //�����ֿ�ĳ���
#define DATANUM (SUBDATANUM * MAX_THREADS)  //64M������

float rawFloatData[DATANUM];   //ԭʼ��������
float result[DATANUM];



int main() {
	LARGE_INTEGER frequency;
	LARGE_INTEGER start, end;
	double elapsedTime1;
	double elapsedTime2;
	QueryPerformanceFrequency(&frequency);

	//���ݳ�ʼ��
	std::cout << "Test Begin!\n";
	for (size_t i = 0; i < DATANUM; i++)
	{
		rawFloatData[i] = float(i + 1);
	}

	float floatSum1 = 0.0;
	{
		QueryPerformanceCounter(&start);//start  
		//�������
		floatSum1 = Sum(rawFloatData, DATANUM);

		QueryPerformanceCounter(&end);//end

		std::cout << std::fixed << std::setprecision(3);
		std::cout << "Sum result  is�� " << floatSum1 << std::endl;
		elapsedTime1 = static_cast<double>(end.QuadPart - start.QuadPart) / frequency.QuadPart;
		std::cout << "Time Consume1:\t" << elapsedTime1 << endl;


		float floatSum2 = 0.0;
		QueryPerformanceCounter(&start);//start  

		//�������
		floatSum2 = sumSpeedUp(rawFloatData, DATANUM);

		QueryPerformanceCounter(&end);//end
		std::cout << std::fixed << std::setprecision(3);
		std::cout << "Speedup Sum result  is�� " << floatSum2 << std::endl;
		elapsedTime2 = static_cast<double>(end.QuadPart - start.QuadPart) / frequency.QuadPart;
		std::cout << "Time Consume1:\t" << elapsedTime2 << endl;
		std::cout << "���ٱ�:\t" << elapsedTime1 / elapsedTime2 << endl;

		float std_float = 1130722617.7269; //python�߾��ȵĽ��ֱ�Ӹ���float������float�ܱ�ʾ���ȷ��ֵ
		std::cout << "��׼���(python�������float)�� " << std_float << std::endl;

	}
	std::cout << endl;
	std::cout << endl;
	{
		QueryPerformanceCounter(&start);//start  
		//�������ֵ
		float floatmax1 = Max(rawFloatData, DATANUM);

		QueryPerformanceCounter(&end);//end

		std::cout << std::fixed << std::setprecision(3);
		std::cout << "Max result  is�� " << floatmax1 << std::endl;
		elapsedTime1 = static_cast<double>(end.QuadPart - start.QuadPart) / frequency.QuadPart;
		std::cout << "Time Consume1:\t" << elapsedTime1 << endl;


		float floatmax2 = 0.0;
		QueryPerformanceCounter(&start);//start  

		//�������
		floatmax2 = maxSpeedUp(rawFloatData, DATANUM);

		QueryPerformanceCounter(&end);//end
		std::cout << std::fixed << std::setprecision(3);
		std::cout << "Speedup Max result  is�� " << floatmax2 << std::endl;
		elapsedTime2 = static_cast<double>(end.QuadPart - start.QuadPart) / frequency.QuadPart;
		std::cout << "Time Consume1:\t" << elapsedTime2 << endl;

		std::cout << "���ٱ�:\t" << elapsedTime1 / elapsedTime2 << endl;
	}
	std::cout << endl;
	std::cout << endl;
	{
		//��������
		//shuffleArray(rawFloatData, DATANUM);

		// ��ȡ��ʼʱ��
		QueryPerformanceCounter(&start);

		// ���򲢲���ʱ��
		Sort(rawFloatData, DATANUM, result);

		// ��ȡ����ʱ��
		QueryPerformanceCounter(&end);

		// ����ʱ������Ϊ��λ��
		elapsedTime1 = static_cast<double>(end.QuadPart - start.QuadPart) / frequency.QuadPart;
		std::cout << "Total Time Consumed: " << elapsedTime1 << " seconds." << std::endl;

		//��������Ƿ���ȷ
		int flag = 1;
		for (int i = 0; i < DATANUM - 1; i++)
			if (result[i] > result[i + 1])
			{
				std::cout << "�������δͨ��..." << std::endl;
				flag = false;
				break;
			}

		if (flag)
			std::cout << "�������ͨ��..." << std::endl;


		// ��ȡ��ʼʱ��
		QueryPerformanceCounter(&start);

		//��������
		sortSpeedUp(rawFloatData, DATANUM, result);

		// ��ȡ����ʱ��
		QueryPerformanceCounter(&end);

		// ����ʱ������Ϊ��λ��
		elapsedTime2 = static_cast<double>(end.QuadPart - start.QuadPart) / frequency.QuadPart;
		std::cout << "Total Time Consumed: " << elapsedTime2 << " seconds." << std::endl;

		//��������Ƿ���ȷ
		flag = 1;
		for (int i = 0; i < DATANUM - 1; i++)
			if (result[i] > result[i + 1])
			{
				std::cout << "�������δͨ��..." << std::endl;
				flag = false;
				break;
			}
		if (flag)
			std::cout << "�������ͨ��..." << std::endl;

		std::cout << "���ٱ�:\t" << elapsedTime1 / elapsedTime2 << endl;
	}

	return 0;
}

