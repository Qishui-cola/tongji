//TCP server  汇总单机加速和不加速的文件
#define _WINSOCK_DEPRECATED_NO_WARNINGS

#pragma comment(lib,"ws2_32.lib")
#include <WinSock2.h>
#include <iostream>
#include <iomanip>  // 包含控制格式化的头文件
#include <immintrin.h> // SSE 指令集
#include <omp.h>       // 包含 OpenMP 头文件
#include <cstring>
#include <sstream>

using namespace std;



//求和
float sumSpeedUp(const float data[], int len);
float avx_reduce(__m256 vec);
//最大
float maxSpeedUp(const float data[], int len);
//排序
void sortSpeedUp(const float data[], const int len, float result[]);
void avx_memcpy(float* dest, const float* src, size_t n);
void quickSortSpeedUp(float* arr, int left, int right);
void merge(float* data, int start1, int end1, int start2, int end2, float* temp);
void parallelMergeSort(float* data, int blockSize, int numBlocks, float* temp);

void shuffleArray(float arr[], int len);



#define MAX_THREADS 64
#define SUBDATANUM 1000*1000   //数据字块的长度
#define DATANUM (SUBDATANUM * MAX_THREADS)  //64M个数据

//float rawFloatData[DATANUM];   //原始数据数组
//float result[DATANUM];
float temp[DATANUM];

//加速求和+Kahan
float avx_reduce(__m256 vec) {
	__m128 hi = _mm256_extractf128_ps(vec, 1);
	__m128 lo = _mm256_castps256_ps128(vec);
	__m128 sum = _mm_add_ps(lo, hi); // 合并高位和低位
	sum = _mm_hadd_ps(sum, sum);    // 水平加法
	sum = _mm_hadd_ps(sum, sum);    // 最终求和
	return _mm_cvtss_f32(sum);      // 提取结果
}

float sumSpeedUp(const float data[], int len) {
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

//加速最大
float maxSpeedUp(const float data[], int len) {
	float max_val = -FLT_MAX;

	// 并行计算
#pragma omp parallel reduction(max:max_val)
	{
		__m256 max_vec = _mm256_set1_ps(-FLT_MAX);  // 初始化为最小值

#pragma omp for schedule(static)
		for (int i = 0; i < len; i += 8) {
			__m256 vec = _mm256_loadu_ps(&data[i]);  // 加载数据
			__m256 sqrt_vec = _mm256_sqrt_ps(vec);   // 平方根
			__m256 log_vec = _mm256_log_ps(sqrt_vec);  // 对数

			// 块内归约
			max_vec = _mm256_max_ps(max_vec, log_vec);
		}

		// 将 AVX 寄存器内的最大值归约到一个标量值
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

//加速排序
void avx_memcpy(float* dest, const float* src, size_t n) {
	size_t i = 0;
	size_t simd_width = 8; // AVX 每次处理 8 个 float 元素

	// 使用 AVX 批量拷贝
	for (; i + simd_width <= n; i += simd_width) {
		__m256 data = _mm256_loadu_ps(src + i);  // 安全加载 8 个 float 元素
		_mm256_storeu_ps(dest + i, data);        // 存储 8 个 float 元素
	}

	// 剩余不足 8 个的部分，逐一处理
	for (; i < n; ++i) {
		dest[i] = src[i];
	}
}

void quickSortSpeedUp(float* arr, int left, int right) {
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

		// 三数取中法选择基准值
		int mid = (left + right) / 2;
		if (arr[left] > arr[right]) std::swap(arr[left], arr[right]);
		if (arr[left] > arr[mid]) std::swap(arr[left], arr[mid]);
		if (arr[mid] > arr[right]) std::swap(arr[mid], arr[right]);
		float pivot = arr[mid];

		// 分区
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

		// 尾递归优化：对较小的一边递归，较大的一边循环处理
		if (j - left < right - i) {
			quickSortSpeedUp(arr, left, j);
			left = i;
		}
		else {
			quickSortSpeedUp(arr, i, right);
			right = j;
		}
	}
}

// 归并函数（合并两个有序段）
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
	//avx_memcpy(data + start1, temp, (end2 - start1 + 1));
}

// 多线程归并排序函数
void parallelMergeSort(float* data, int blockSize, int numBlocks, float* temp) {
	//int numThreads = 16;

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

// 主排序函数
void sortSpeedUp(const float data[], const int len, float result[]) {
	// 将原始数据复制到 result 数组
	/*for (int i = 0; i < len; ++i) {
		result[i] = log(sqrt(data[i]));
	}*/
	// 将原始数据复制到 result 数组 AVX 加速
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

			// 加载数据到 AVX 寄存器
			input = _mm256_loadu_ps(&data[i]);

			// 计算 sqrt 和 log
			sqrt_result = _mm256_sqrt_ps(input);
			log_result = _mm256_log_ps(sqrt_result);

			// 存储计算结果
			_mm256_storeu_ps(&result[i], log_result);

			// 对于剩余不足 8 个的元素，单独处理
			for (int j = 0; j < remaining; ++j) {
				result[i + j] = log(sqrt(data[i + j]));
			}
		}
	}

// 使用快速排序对 result 数组进行排序
//quickSort(result, 0, len - 1);
//分块并行
const int BLOCK = DATANUM / MAX_THREADS;
#pragma omp parallel num_threads(64) //必须是64
{
	int thread_id = omp_get_thread_num();
	quickSortSpeedUp(result, thread_id * BLOCK, (thread_id + 1) * BLOCK - 1);
	//cout << int(thread_id * DATANUM / MAX_THREADS) << "   " << int((thread_id + 1) * DATANUM / MAX_THREADS - 1) << endl;
}

//归并分块
parallelMergeSort(result, BLOCK, 64, temp);
}

//打乱函数
void shuffleArray(float arr[], int len) {
	// 设置随机种子
	std::srand(std::time(0));

	for (int i = len - 1; i > 0; --i) {
		// 生成 [0, i] 范围内的随机索引
		int j = std::rand() % (i + 1);

		// 交换 arr[i] 和 arr[j]
		std::swap(arr[i], arr[j]);
	}
}



