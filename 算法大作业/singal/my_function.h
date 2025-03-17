#pragma once

#include <iomanip>  // �������Ƹ�ʽ����ͷ�ļ�
#include <immintrin.h> // SSE ָ�
#include <omp.h>       // ���� OpenMP ͷ�ļ�

//�������㷨
float Sum(const float data[], int len);
float Max(const float data[], int len);
void quickSort(float* arr, int left, int right);
void Sort(const float data[], const int len, float result[]);

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
