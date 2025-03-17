//TCP server  ���ܵ������ٺͲ����ٵ��ļ�
#define _WINSOCK_DEPRECATED_NO_WARNINGS

#pragma comment(lib,"ws2_32.lib")
#include <WinSock2.h>
#include <iostream>
#include <iomanip>  // �������Ƹ�ʽ����ͷ�ļ�
#include <immintrin.h> // SSE ָ�
#include <omp.h>       // ���� OpenMP ͷ�ļ�
#include <cstring>
#include <sstream>

using namespace std;



//���
float sumSpeedUp(const float data[], int len);
float avx_reduce(__m256 vec);
//���
float maxSpeedUp(const float data[], int len);
//����
void sortSpeedUp(const float data[], const int len, float result[]);
void avx_memcpy(float* dest, const float* src, size_t n);
void quickSortSpeedUp(float* arr, int left, int right);
void merge(float* data, int start1, int end1, int start2, int end2, float* temp);
void parallelMergeSort(float* data, int blockSize, int numBlocks, float* temp);

void shuffleArray(float arr[], int len);



#define MAX_THREADS 64
#define SUBDATANUM 1000*1000   //�����ֿ�ĳ���
#define DATANUM (SUBDATANUM * MAX_THREADS)  //64M������

//float rawFloatData[DATANUM];   //ԭʼ��������
//float result[DATANUM];
float temp[DATANUM];

//�������+Kahan
float avx_reduce(__m256 vec) {
	__m128 hi = _mm256_extractf128_ps(vec, 1);
	__m128 lo = _mm256_castps256_ps128(vec);
	__m128 sum = _mm_add_ps(lo, hi); // �ϲ���λ�͵�λ
	sum = _mm_hadd_ps(sum, sum);    // ˮƽ�ӷ�
	sum = _mm_hadd_ps(sum, sum);    // �������
	return _mm_cvtss_f32(sum);      // ��ȡ���
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

//�������
float maxSpeedUp(const float data[], int len) {
	float max_val = -FLT_MAX;

	// ���м���
#pragma omp parallel reduction(max:max_val)
	{
		__m256 max_vec = _mm256_set1_ps(-FLT_MAX);  // ��ʼ��Ϊ��Сֵ

#pragma omp for schedule(static)
		for (int i = 0; i < len; i += 8) {
			__m256 vec = _mm256_loadu_ps(&data[i]);  // ��������
			__m256 sqrt_vec = _mm256_sqrt_ps(vec);   // ƽ����
			__m256 log_vec = _mm256_log_ps(sqrt_vec);  // ����

			// ���ڹ�Լ
			max_vec = _mm256_max_ps(max_vec, log_vec);
		}

		// �� AVX �Ĵ����ڵ����ֵ��Լ��һ������ֵ
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

//��������
void avx_memcpy(float* dest, const float* src, size_t n) {
	size_t i = 0;
	size_t simd_width = 8; // AVX ÿ�δ��� 8 �� float Ԫ��

	// ʹ�� AVX ��������
	for (; i + simd_width <= n; i += simd_width) {
		__m256 data = _mm256_loadu_ps(src + i);  // ��ȫ���� 8 �� float Ԫ��
		_mm256_storeu_ps(dest + i, data);        // �洢 8 �� float Ԫ��
	}

	// ʣ�಻�� 8 ���Ĳ��֣���һ����
	for (; i < n; ++i) {
		dest[i] = src[i];
	}
}

void quickSortSpeedUp(float* arr, int left, int right) {
	while (left < right) {
		if (right - left <= 16) { // С��ģ�Ż�����������
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

		// ����ȡ�з�ѡ���׼ֵ
		int mid = (left + right) / 2;
		if (arr[left] > arr[right]) std::swap(arr[left], arr[right]);
		if (arr[left] > arr[mid]) std::swap(arr[left], arr[mid]);
		if (arr[mid] > arr[right]) std::swap(arr[mid], arr[right]);
		float pivot = arr[mid];

		// ����
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

		// β�ݹ��Ż����Խ�С��һ�ߵݹ飬�ϴ��һ��ѭ������
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

// �鲢�������ϲ���������Σ�
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

// ���̹߳鲢������
void parallelMergeSort(float* data, int blockSize, int numBlocks, float* temp) {
	//int numThreads = 16;

	while (numBlocks > 1) {
		int newNumBlocks = numBlocks / 2; // ÿ�ι鲢����һ�����
#pragma omp parallel for //num_threads(numThreads)  //���̴߳���ע��������ܲ��絥�߳�
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

// ��������
void sortSpeedUp(const float data[], const int len, float result[]) {
	// ��ԭʼ���ݸ��Ƶ� result ����
	/*for (int i = 0; i < len; ++i) {
		result[i] = log(sqrt(data[i]));
	}*/
	// ��ԭʼ���ݸ��Ƶ� result ���� AVX ����
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

			// �������ݵ� AVX �Ĵ���
			input = _mm256_loadu_ps(&data[i]);

			// ���� sqrt �� log
			sqrt_result = _mm256_sqrt_ps(input);
			log_result = _mm256_log_ps(sqrt_result);

			// �洢������
			_mm256_storeu_ps(&result[i], log_result);

			// ����ʣ�಻�� 8 ����Ԫ�أ���������
			for (int j = 0; j < remaining; ++j) {
				result[i + j] = log(sqrt(data[i + j]));
			}
		}
	}

// ʹ�ÿ�������� result �����������
//quickSort(result, 0, len - 1);
//�ֿ鲢��
const int BLOCK = DATANUM / MAX_THREADS;
#pragma omp parallel num_threads(64) //������64
{
	int thread_id = omp_get_thread_num();
	quickSortSpeedUp(result, thread_id * BLOCK, (thread_id + 1) * BLOCK - 1);
	//cout << int(thread_id * DATANUM / MAX_THREADS) << "   " << int((thread_id + 1) * DATANUM / MAX_THREADS - 1) << endl;
}

//�鲢�ֿ�
parallelMergeSort(result, BLOCK, 64, temp);
}

//���Һ���
void shuffleArray(float arr[], int len) {
	// �����������
	std::srand(std::time(0));

	for (int i = len - 1; i > 0; --i) {
		// ���� [0, i] ��Χ�ڵ��������
		int j = std::rand() % (i + 1);

		// ���� arr[i] �� arr[j]
		std::swap(arr[i], arr[j]);
	}
}



