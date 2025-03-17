//TCP server
#define _WINSOCK_DEPRECATED_NO_WARNINGS
#pragma comment(lib,"ws2_32.lib")
#include <WinSock2.h>
#include <iostream>
#include <string>
#include <immintrin.h> // SSE 指令集
#include <omp.h>       // 包含 OpenMP 头文件
#include <cstring>
#include <sstream>
#include "zlib.h"
#include "my_function.h"

using namespace std;

typedef unsigned int uint32;

#define MAX_THREADS 64
#define SUBDATANUM 1000*1000 //数据字块的长度
#define DATANUM (SUBDATANUM * MAX_THREADS)  //64M个数据
#define RECEIVEONCE DATANUM

float rawFloatData[DATANUM];   //原始数据数组
float result[DATANUM];
float recv_result[DATANUM];
float Finalresult[DATANUM * 2];

float compressedData[DATANUM + 20];


int ThreadID[MAX_THREADS];
float floatResults[MAX_THREADS];//每个线程的中间结果
HANDLE hSemaphores[MAX_THREADS];//信号量，保证不重入。等同于mutex
#define REPEATTIMES 5


//将传输来的数组和自己的归并
void mergeArrays(float arr1[], int size1, float arr2[], int size2, float mergedArray[]) {
	int i = 0; // 指向 arr1 的当前索引
	int j = 0; // 指向 arr2 的当前索引
	int k = 0; // 指向 mergedArray 的当前索引

	// 当两个数组都有剩余元素时，选择较小的元素放入 mergedArray
	while (i < size1 && j < size2) {
		if (arr1[i] <= arr2[j]) {
			mergedArray[k++] = arr1[i++];
		}
		else {
			mergedArray[k++] = arr2[j++];
		}
	}

	// 如果 arr1 中有剩余元素，全部放入 mergedArray
	while (i < size1) {
		mergedArray[k++] = arr1[i++];
	}

	// 如果 arr2 中有剩余元素，全部放入 mergedArray
	while (j < size2) {
		mergedArray[k++] = arr2[j++];
	}
}

// 解压函数
unsigned char* decompressData(const unsigned char* compressedData, size_t compressedSize, size_t originalSize) {
	unsigned char* decompressedData = new unsigned char[originalSize];

	uLongf decompressedSize = originalSize;
	int result = uncompress(decompressedData, &decompressedSize, compressedData, compressedSize);
	if (result != Z_OK) {
		delete[] decompressedData;
		throw std::runtime_error("Decompression failed!");
	}

	return decompressedData;
}

//MD5类
class MD5 {
public:
	MD5() { init(); }

	void update(const unsigned char* input, size_t length) {
		size_t index = _count[0] / 8 % 64;
		_count[0] += static_cast<uint32>(length << 3);
		if (_count[0] < (length << 3)) _count[1]++;
		_count[1] += static_cast<uint32>(length >> 29);

		size_t firstPart = 64 - index;
		size_t i;

		if (length >= firstPart) {
			std::memcpy(&_buffer[index], input, firstPart);
			transform(_buffer);

			for (i = firstPart; i + 63 < length; i += 64)
				transform(&input[i]);

			index = 0;
		}
		else {
			i = 0;
		}
		std::memcpy(&_buffer[index], &input[i], length - i);
	}

	std::string final() {
		static unsigned char PADDING[64] = { 0x80 };
		unsigned char bits[8];
		encode(bits, _count, 8);

		size_t index = _count[0] / 8 % 64;
		size_t padLen = (index < 56) ? (56 - index) : (120 - index);
		update(PADDING, padLen);
		update(bits, 8);

		unsigned char digest[16];
		encode(digest, _state, 16);

		std::ostringstream result;
		for (int i = 0; i < 16; i++)
			result << std::hex << std::setw(2) << std::setfill('0') << (int)digest[i];

		init();
		return result.str();
	}

	void update(const std::string& str) {
		update(reinterpret_cast<const unsigned char*>(str.c_str()), str.size());
	}

private:
	void init() {
		_count[0] = _count[1] = 0;
		_state[0] = 0x67452301;
		_state[1] = 0xefcdab89;
		_state[2] = 0x98badcfe;
		_state[3] = 0x10325476;
	}

	void transform(const unsigned char block[64]) {
		static const uint32 S[4][4] = {
			{7, 12, 17, 22}, {5, 9, 14, 20}, {4, 11, 16, 23}, {6, 10, 15, 21}
		};
		static const uint32 K[64] = { /* 初始化常量表，省略内容 */ };

		uint32 a = _state[0], b = _state[1], c = _state[2], d = _state[3];
		uint32 X[16];
		decode(X, block, 64);

		for (uint32 i = 0; i < 64; i++) {
			uint32 F, g;
			if (i < 16) {
				F = (b & c) | ((~b) & d);
				g = i;
			}
			else if (i < 32) {
				F = (d & b) | ((~d) & c);
				g = (5 * i + 1) % 16;
			}
			else if (i < 48) {
				F = b ^ c ^ d;
				g = (3 * i + 5) % 16;
			}
			else {
				F = c ^ (b | (~d));
				g = (7 * i) % 16;
			}
			F += a + K[i] + X[g];
			a = d;
			d = c;
			c = b;
			b += (F << S[i / 16][i % 4]) | (F >> (32 - S[i / 16][i % 4]));
		}

		_state[0] += a;
		_state[1] += b;
		_state[2] += c;
		_state[3] += d;
	}

	void encode(unsigned char* output, const uint32* input, size_t length) {
		for (size_t i = 0; i < length / 4; i++) {
			output[i * 4] = input[i] & 0xff;
			output[i * 4 + 1] = (input[i] >> 8) & 0xff;
			output[i * 4 + 2] = (input[i] >> 16) & 0xff;
			output[i * 4 + 3] = (input[i] >> 24) & 0xff;
		}
	}

	void decode(uint32* output, const unsigned char* input, size_t length) {
		for (size_t i = 0; i < length / 4; i++) {
			output[i] = input[i * 4] | (input[i * 4 + 1] << 8) |
				(input[i * 4 + 2] << 16) | (input[i * 4 + 3] << 24);
		}
	}

	uint32 _state[4], _count[2];
	unsigned char _buffer[64];
};

int main()
{
	LARGE_INTEGER frequency;
	LARGE_INTEGER start, end;
	double elapsed1;

	QueryPerformanceFrequency(&frequency);

	//数据初始化
	std::cout << "Hello World!\n";
	for (size_t i = 0; i < DATANUM; i++)
	{
		rawFloatData[i] = float(i + 1);
		recv_result[i] = float(i + 1) + 64 * 1000 * 1000;
	}
	for (size_t i = 0; i < DATANUM; i++)
	{
		recv_result[i] = log(sqrt(recv_result[i]));
	}

	//启动sock
	WSAData wsaData;
	WORD DllVersion = MAKEWORD(2, 1);
	if (WSAStartup(DllVersion, &wsaData) != 0)
	{
		MessageBoxA(NULL, "WinSock startup error", "Error", MB_OK | MB_ICONERROR);
		return 0;
	}

	SOCKADDR_IN addr;
	int addrlen = sizeof(addr);
	addr.sin_family = AF_INET; //IPv4 Socket
	addr.sin_port = htons(8888); // sever Port
	addr.sin_addr.s_addr = inet_addr("172.20.10.6"); //server PC

	SOCKET sListen = socket(AF_INET, SOCK_STREAM, NULL);//公司的接线员，不是处理业务的人 
	bind(sListen, (SOCKADDR*)&addr, sizeof(addr)); //服务器必须绑定

	listen(sListen, SOMAXCONN); //接线员聆听是否有生意

	//while ()
	SOCKET newConnection; //真正的业务员  build a new socket do new connection. the sListen is just listenning not used to exchange data
	newConnection = accept(sListen, (SOCKADDR*)&addr, &addrlen);

	if (newConnection == 0)
	{
		std::cout << "bad connection." << std::endl;
	}
	else
	{
		cout << "begin" << endl;

		float floatSum1 = 0.0;
		float floatSum2 = 0.0;
		float floatMax1 = 0.0;
		float floatMax2 = 0.0;

		//计算求和
		QueryPerformanceCounter(&start);//start  
		floatSum1 = sumSpeedUp(rawFloatData, DATANUM);
		//cout << floatSum1 << endl;
		recv(newConnection, (char*)&floatSum2, sizeof(floatSum2), NULL);
		floatSum1 += floatSum2;
		QueryPerformanceCounter(&end);//end

		elapsed1 = static_cast<double>(end.QuadPart - start.QuadPart) / frequency.QuadPart;
		std::cout << "AVX + OPENMP Time Consume1:\t" << elapsed1 << endl;
		std::cout << "result from client is： " << floatSum2 << std::endl;
		std::cout << "all result from is： " << floatSum1 << std::endl;

		cout << endl;
		cout << endl;

		//计算最大
		QueryPerformanceCounter(&start);//start  

		floatMax1 = maxSpeedUp(rawFloatData, DATANUM);
		recv(newConnection, (char*)&floatMax2, sizeof(floatMax2), NULL);

		floatMax1 = (floatMax1 > floatMax2 ? floatMax1 : floatMax2);
		QueryPerformanceCounter(&end);//end

		elapsed1 = static_cast<double>(end.QuadPart - start.QuadPart) / frequency.QuadPart;
		std::cout << "AVX + OPENMP Time Consume1:\t" << elapsed1 << endl;
		std::cout << "result from client is： " << floatMax2 << std::endl;
		std::cout << "max result from is： " << floatMax1 << std::endl;

		cout << endl;
		cout << endl;

		////计算排序
		QueryPerformanceCounter(&start);//start  
		//接收压缩数据大小
		size_t compressedSize = 0;
		recv(newConnection, (char*)&compressedSize, sizeof(compressedSize), NULL);
		//cout << "接收数据量：" << compressedSize << endl;

		//接收MD5码
		char md5recv[40];
		recv(newConnection, (char*)&md5recv,32, NULL);
		md5recv[32] = '\0';
		//std::cout << "MD5: " << md5recv << std::endl;

		//-------------------------------------------------------------------------------------------------------分块模块

		int receivesuccess = 0;//记录一次接收值的数量
		int received = 0;//累加存储接收的数据的数量
		int i = 0;
		char* receivePin = (char*)compressedData;//数组起始

		while (1) {
			// 尝试接收数据
			receivesuccess = recv(newConnection, &receivePin[received], compressedSize / 16, 0);
			// 打印接收结果
			//printf("第%d次接收：接收数据量：%d\n", i, receivesuccess);
			//i++;

			if (receivesuccess > 0) {
				// 接收成功，累加接收到的数据量
				received += receivesuccess;
			}
			else if (receivesuccess == 0) {
				// 连接被关闭
				printf("Connection closed by the peer.\n");
				break;
			}
			else {
				// 接收失败，打印错误信息
				int erron = WSAGetLastError();
				printf("Error in recv: %d\n", erron);
				break;
			}

			// 检查是否接收完所有数据
			if (received >= compressedSize) {
				break;
			}
		}

		////------------------------------------------------------------------------------------------------------------
		//cout << received << endl;

		sortSpeedUp(rawFloatData, DATANUM, result);

		//解压数据
		// 解压数据
		unsigned char* decompressedData = decompressData((unsigned char*)compressedData, compressedSize, DATANUM * 4);
		float* receiveSorts = (float*)decompressedData;
		//归并数组
		mergeArrays(receiveSorts, DATANUM, result, DATANUM, Finalresult);

		QueryPerformanceCounter(&end);//end

		elapsed1 = static_cast<double>(end.QuadPart - start.QuadPart) / frequency.QuadPart;
		std::cout << "AVX + OPENMP Time Consume1:\t" << elapsed1 << endl;
		

		//检测数据是否正确
		// 示例输入数组 MD5比较
		MD5 md5;
		md5.update((const unsigned char*)compressedData, compressedSize);
		std::string md5Result = md5.final();
		//std::cout << "MD5: " << md5Result << std::endl;

		for (int i = 0; i < md5Result.size(); i++) {
			if (md5Result[i] != md5recv[i]) {
				cout << "传输出错" << endl;
				break;
			}
			if (i == md5Result.size() - 1)
				cout << "传输成功" << endl;
		}

		//测试排序是否正确
		int flag = 1;
		for (int i = 0; i < 2 * DATANUM - 1; i++)
			if ((Finalresult[i] > Finalresult[i + 1]))
			{
				std::cout << "排序测试未通过..." << std::endl;
				std::cout << i << "  " << Finalresult[i] << "  " << Finalresult[i + 1] << endl;
				flag = false;
				break;
			}
		if (flag)
			std::cout << "排序测试通过..." << std::endl;

	}

	cout << "按 Enter 继续..." << endl;
	cin.get();

	//cleanup
	closesocket(newConnection);
	WSACleanup();
	return 0;
}

