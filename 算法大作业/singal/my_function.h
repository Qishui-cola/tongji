#pragma once

#include <iomanip>  // 包含控制格式化的头文件
#include <immintrin.h> // SSE 指令集
#include <omp.h>       // 包含 OpenMP 头文件

//不加速算法
float Sum(const float data[], int len);
float Max(const float data[], int len);
void quickSort(float* arr, int left, int right);
void Sort(const float data[], const int len, float result[]);

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
