//TCP client

#define _WINSOCK_DEPRECATED_NO_WARNINGS

#pragma comment(lib,"ws2_32.lib")
#include <WinSock2.h>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h> // 包含 SSE 指令头文件
#include <cmath>
#include <cstring>
#include <sstream>
#include <iomanip>
#include "zlib.h"

using namespace std;

typedef unsigned int uint32;

#define MAX_THREADS 64
#define SUBDATANUM 1000*1000
#define DATANUM (SUBDATANUM * MAX_THREADS)
#include <omp.h>
int rawIntData[DATANUM];//shared by son-threads
float rawFloatData[DATANUM];   //原始数据数组
float floatResults[DATANUM / 8];//每个线程的中间结果
int ThreadID[MAX_THREADS];
float result[DATANUM];
float temp[DATANUM];

HANDLE hSemaphores[MAX_THREADS];//信号量，保证不重入。等同于mutex
#define REPEATTIMES 5



 //压缩函数
unsigned char* compressData(const unsigned char* data, size_t dataSize, size_t& compressedSize) {
	uLongf maxCompressedSize = compressBound(dataSize);
	unsigned char* compressedData = new unsigned char[maxCompressedSize];

	int result = compress(compressedData, &maxCompressedSize, data, dataSize);
	if (result != Z_OK) {
		delete[] compressedData;
		throw std::runtime_error("Compression failed!");
	}

	compressedSize = maxCompressedSize;
	return compressedData;
}

//打乱函数
void shuffleArray(float arr[], int len) {
	 //设置随机种子
	std::srand(std::time(0));

	for (int i = len - 1; i > 0; --i) {
		 //生成 [0, i] 范围内的随机索引
		int j = std::rand() % (i + 1);

		 //交换 arr[i] 和 arr[j]
		std::swap(arr[i], arr[j]);
	}
}

float avx_reduce(__m256 vec) {
	__m128 hi = _mm256_extractf128_ps(vec, 1);
	__m128 lo = _mm256_castps256_ps128(vec);
	__m128 sum = _mm_add_ps(lo, hi); // 合并高位和低位
	sum = _mm_hadd_ps(sum, sum);    // 水平加法
	sum = _mm_hadd_ps(sum, sum);    // 最终求和
	return _mm_cvtss_f32(sum);      // 提取结果
}

float Sum(const float data[], int len) {
	float final_sum = 0.0f;

#pragma omp parallel
	{
		__m256 local_sum = _mm256_setzero_ps();
		__m256 local_correction = _mm256_setzero_ps();

#pragma omp for schedule(static)
		for (int i = 0; i < len; i += 8) {
			__m256 vec = _mm256_loadu_ps(&data[i]);
			__m256 sqrt_vec = _mm256_sqrt_ps(vec);
			__m256 log_vec = _mm256_log_ps(sqrt_vec);

			__m256 y = _mm256_sub_ps(log_vec, local_correction);
			__m256 t = _mm256_add_ps(local_sum, y);
			local_correction = _mm256_sub_ps(_mm256_sub_ps(t, local_sum), y);
			local_sum = t;
		}

		float thread_sum = avx_reduce(local_sum);

#pragma omp atomic
		final_sum += thread_sum;
	}

	return final_sum;
}

float Max(const float data[], int len) {
	float max_val = -FLT_MAX;

	 //并行计算
#pragma omp parallel reduction(max:max_val)
	{
		__m256 max_vec = _mm256_set1_ps(-FLT_MAX);  // 初始化为最小值

#pragma omp for schedule(static)
		for (int i = 0; i < len; i += 8) {
			__m256 vec = _mm256_loadu_ps(&data[i]);  // 加载数据
			__m256 sqrt_vec = _mm256_sqrt_ps(vec);   // 平方根
			__m256 log_vec = _mm256_log_ps(sqrt_vec);  // 对数

			 //块内归约
			max_vec = _mm256_max_ps(max_vec, log_vec);
		}

		 //将 AVX 寄存器内的最大值归约到一个标量值
		alignas(32) float temp[8];
		_mm256_store_ps(temp, max_vec);
		for (int j = 0; j < 8; ++j) {
			if (temp[j] > max_val) {
				max_val = temp[j];
			}
		}
	}

	return max_val;
}

void avx_memcpy(float* dest, const float* src, size_t n) {
	size_t i = 0;
	size_t simd_width = 8; // AVX 每次处理 8 个 float 元素

	 //使用 AVX 批量拷贝
	for (; i + simd_width <= n; i += simd_width) {
		__m256 data = _mm256_loadu_ps(src + i);  // 安全加载 8 个 float 元素
		_mm256_storeu_ps(dest + i, data);        // 存储 8 个 float 元素
	}

	 //剩余不足 8 个的部分，逐一处理
	for (; i < n; ++i) {
		dest[i] = src[i];
	}
}

void quickSort(float* arr, int left, int right) {
	while (left < right) {
		if (right - left <= 16) { // 小规模优化：插入排序
			for (int k = left + 1; k <= right; k++) {
				float temp = arr[k];
				int l = k - 1;
				while (l >= left && arr[l] > temp) {
					arr[l + 1] = arr[l];
					l--;
				}
				arr[l + 1] = temp;
			}
			return;
		}

		 //三数取中法选择基准值
		int mid = (left + right) / 2;
		if (arr[left] > arr[right]) std::swap(arr[left], arr[right]);
		if (arr[left] > arr[mid]) std::swap(arr[left], arr[mid]);
		if (arr[mid] > arr[right]) std::swap(arr[mid], arr[right]);
		float pivot = arr[mid];

		 //分区
		int i = left, j = right;
		while (i <= j) {
			while (arr[i] < pivot) i++;
			while (arr[j] > pivot) j--;
			if (i <= j) {
				std::swap(arr[i], arr[j]);
				i++;
				j--;
			}
		}

		//尾递归优化：对较小的一边递归，较大的一边循环处理
		if (j - left < right - i) {
			quickSort(arr, left, j);
			left = i;
		}
		else {
			quickSort(arr, i, right);
			right = j;
		}
	}
}

//分块归并函数
 //归并函数（合并两个有序段）
void merge(float* data, int start1, int end1, int start2, int end2, float* temp) {
	int i = start1, j = start2, k = 0;
	while (i <= end1 && j <= end2) {
		if (data[i] <= data[j]) {
			temp[k++] = data[i++];
		}
		else {
			temp[k++] = data[j++];
		}
	}
	while (i <= end1) {
		temp[k++] = data[i++];
	}
	while (j <= end2) {
		temp[k++] = data[j++];
	}
	std::memcpy(data + start1, temp, sizeof(float) * (end2 - start1 + 1));
	avx_memcpy(data + start1, temp, (end2 - start1 + 1));
}

//多线程归并排序函数
void parallelMergeSort(float* data, int blockSize, int numBlocks, float* temp) {
	int numThreads = 16;

	while (numBlocks > 1) {
		int newNumBlocks = numBlocks / 2; // 每次归并减少一半块数
#pragma omp parallel for //num_threads(numThreads)  //多线程处理，注意后续可能不如单线程
		for (int i = 0; i < newNumBlocks; i++) {
			int start1 = i * 2 * blockSize;
			int end1 = min(start1 + blockSize - 1, (numBlocks * blockSize) - 1);
			int start2 = end1 + 1;
			int end2 = min(start2 + blockSize - 1, (numBlocks * blockSize) - 1);

			if (start2 < (numBlocks * blockSize)) {
				merge(data, start1, end1, start2, end2, temp + start1);
			}
		}
		blockSize *= 2;
		numBlocks = newNumBlocks;
	}
}

 //主排序函数
void Sort(const float data[], const int len, float result[]) {
	 //将原始数据复制到 result 数组 AVX 加速
#pragma omp parallel num_threads(16)
	{
		int thread_count = omp_get_num_threads();
		int thread_id = omp_get_thread_num();
		int chunk_size = (len + thread_count - 1) / thread_count;

		int start = thread_id * chunk_size;
		int end = (start + chunk_size) < len ? (start + chunk_size) : len;

		for (int i = start; i < end; i += 8) {
			int remaining = 8 < (end - i) ? 8 : (end - i);
			__m256 input, sqrt_result, log_result;

			 //加载数据到 AVX 寄存器
			input = _mm256_loadu_ps(&data[i]);

			 //计算 sqrt 和 log
			sqrt_result = _mm256_sqrt_ps(input);
			log_result = _mm256_log_ps(sqrt_result);

			 //存储计算结果
			_mm256_storeu_ps(&result[i], log_result);

			 //对于剩余不足 8 个的元素，单独处理
			for (int j = 0; j < remaining; ++j) {
				result[i + j] = log(sqrt(data[i + j]));
			}
		}
	}

 //使用快速排序对 result 数组进行排序
quickSort(result, 0, len - 1);
//分块并行
const int BLOCK = DATANUM / MAX_THREADS;
#pragma omp parallel num_threads(64) //必须是64
{
	int thread_id = omp_get_thread_num();
	quickSort(result, thread_id * BLOCK, (thread_id + 1) * BLOCK - 1);
	//cout << int(thread_id * DATANUM / MAX_THREADS) << "   " << int((thread_id + 1) * DATANUM / MAX_THREADS - 1) << endl;
}

//归并分块
parallelMergeSort(result, BLOCK, 64, temp);
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
	//数据初始化
	std::cout << "Hello World!\n";
	for (size_t i = 0; i < DATANUM; i++)
	{
		rawFloatData[i] = float(i + 1)+64*1000*1000;
	}

	//shuffleArray(rawFloatData, DATANUM);

	WSAData wsaData;
	WORD DllVersion = MAKEWORD(2, 1);
	if (WSAStartup(DllVersion, &wsaData) != 0) 
	{
		MessageBoxA(NULL, "Winsock startup error", "Error", MB_OK | MB_ICONERROR);
		exit(1);
	}

	SOCKADDR_IN serverAddr; //Adres przypisany do socketu Connection
	int sizeofaddr = sizeof(serverAddr);
	serverAddr.sin_family = AF_INET; //IPv4 Socket
	serverAddr.sin_addr.s_addr = inet_addr("100.78.40.235"); //服务器的ip地址
	serverAddr.sin_port = htons(8888); //服务器的Port

	SOCKET Connection = socket(AF_INET, SOCK_STREAM, NULL); 
	if (connect(Connection, (SOCKADDR*)&serverAddr, sizeofaddr) != 0) //连接服务器，三次握手
	{
		MessageBoxA(NULL, "Blad Connection", "Error", MB_OK | MB_ICONERROR);
    	return 0;
	}
	else {
		float floatmax = 0.0;
		float floatSum = 0.0;
		int L1, L3;
		L1 = -1;//开始循环

		//recv(Connection, (char*)&rawFloatData, sizeof(rawFloatData), NULL);
		//std::cout << "Is over " << std::endl;

		floatSum = Sum(rawFloatData, DATANUM);
	    send(Connection, (char*)&floatSum, sizeof(floatSum), NULL);
		std::cout << "result is sended   " << floatSum << std::endl;

		floatmax = Max(rawFloatData, DATANUM);
		send(Connection, (char*)&floatmax, sizeof(floatmax), NULL);
		std::cout << "result is sended   " << floatmax << std::endl;

		Sort(rawFloatData, DATANUM, result);
	}
	int flag = 1;
	for (int i = 0; i < DATANUM - 1; i++) 
		if (result[i] > result[i + 1])
		{
			std::cout << "排序测试未通过..." << std::endl;
			flag = false;
			break;
		}
		if (flag) {
			std::cout << "排序测试通过..." << std::endl;

			//压缩
			size_t compressedSize = 0;
			unsigned char* compressedData = compressData(reinterpret_cast<const unsigned char*>(result), DATANUM*4, compressedSize);
			//std::cout << "Compression successful. Original size: " << DATANUM*4
			//	<< ", Compressed size: " << compressedSize << std::endl;
			//压缩结束
			
			//传输压缩大小
			send(Connection, (char*)& compressedSize, sizeof(compressedSize), NULL);

			//传输MD5码
			MD5 md5;
			md5.update((const unsigned char*)compressedData, compressedSize);
			std::string md5Result = md5.final();
			//std::cout << "MD5: " << md5Result <<"      size"<< md5Result.size() << std::endl;

			send(Connection, md5Result.c_str(), md5Result.size(), NULL);

			//分块传输
			int sended;
			int receivesuccess = 0;//记录一次接收值的数量
			int received = 0;//累加存储接收的数据的数量
			int i = 0;
			char* receivePin = (char*)compressedData;//数组起始
			while (1) {

				receivesuccess = send(Connection, &receivePin[received], compressedSize / 16 , NULL);
				//printf("第%d次发送：接收数据量：%d\n", i, receivesuccess);
				//i = i + 1;

				if (receivesuccess == -1)//打印错误信息
				{
				}
				else
				{
					received = received + receivesuccess;
					
				}
				//if (received >= (4 * DATANUM + 2))
				if (received >= compressedSize)
				{
					break;
				}
			}
			//------------------------------------------------------------------------------------------------------------
			//std::cout << received << std::endl;


			std::cout << "over" << std::endl;
			
		}
		
	closesocket(Connection);
	WSACleanup();
	std::cin.get();
	return 0;
}

