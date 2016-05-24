// 2016-5-16 21:31:46
// Stanley Xiao
// cudaIDW��ʵ��

#include "cudaIDWH.h"
#include "cudaIDWCPU.h"
#include "cudaIDWGPU.h"

// �Ƿ�ʹ��GPU Shared Memory
#define _USE_GPU_SHARED_MEMORY_ 0

#pragma warning (disable: 4819)

/*
 * P.S. 
 * 1/ Լ����ϵ(int* constraints)���ڴ˴�����
 * 2/ IDW����ֻ���������
 */

void cudaIDW(
	GeoPoint3D* sampledData,			// DEM����������õ��ĵ㼯
	int sizeOfSampledData,				// �����㼯�ĵ���
	int width,							// DEM���
	int height,							// DEM�߶�
	float* unknownX,					// ����ֵ��X����
	float* unknownY,					// ����ֵ��Y����
	int sizeOfPointsForInterpolation,	// ����ֵ�����
	bool isCudaDriven,					// �Ƿ����CUDA-IDW
	bool useManhanttenDistance,			// �Ƿ����Manhatten����
	float power,						// ������Ȩ
	GeoPoint3D** expandedData,			// ��ֵ��ĵ㼯
	int& sizeOfExpandedData				// ��ֵ��ĵ㼯�ĵ���
	)
{
	// ����CUDA�Զ���ṹ��float3����GeoPoint3D
	float3* knownPoints = new float3[sizeOfSampledData];
	for (int i = 0; i < sizeOfSampledData; i++)
		knownPoints[i] = make_float3(sampledData[i].x, sampledData[i].y, sampledData[i].z);

	// �����������Ĵ�С
	sizeOfExpandedData = sizeOfSampledData + sizeOfPointsForInterpolation;

	// �洢CPU-IDW�㷨�Ĳ�ֵ���
	float3* unknownPoints = new float3[sizeOfPointsForInterpolation];
	// ��ʼ��
	for (int i = 0; i < sizeOfPointsForInterpolation; i++)
		unknownPoints[i] = make_float3(unknownX[i], unknownY[i], -32767.0f);

	// debug ��ʱ
	clock_t time0 = clock();

	if (isCudaDriven)
	{
		// ʹ��CUDA-IDW
#if _USE_GPU_SHARED_MEMORY_
		// ʹ��GPU�����ڴ�
		IDWInGPUWithSM(unknownPoints, sizeOfPointsForInterpolation, knownPoints, sizeOfSampledData, power, useManhanttenDistance);
#else
		// ��ʹ��GPU�����ڴ�
		IDWInGPU(unknownPoints, sizeOfPointsForInterpolation, knownPoints, sizeOfSampledData, power, useManhanttenDistance);
#endif
	}
	else
	{
		// ʹ��CPU-IDW
		IDWInCPU(unknownPoints, sizeOfPointsForInterpolation, knownPoints, sizeOfSampledData, power, useManhanttenDistance);
	}

	// debug ��ʱ
	std::cout << "IDW time = " << (double)(clock() - time0) / CLOCKS_PER_SEC << "s" << std::endl;

	// �洢��ֵ���
	GeoPoint3D* results = new GeoPoint3D[sizeOfExpandedData];
	for (int i = 0; i < sizeOfSampledData; i++)
		results[i] = GeoPoint3D(knownPoints[i].x, knownPoints[i].y, knownPoints[i].z);
	for (int i = sizeOfSampledData, j = 0; i < sizeOfExpandedData; i++, j++)
		results[i] = GeoPoint3D(unknownPoints[j].x, unknownPoints[j].y, unknownPoints[j].z);
	*expandedData = results;

	//delete[] results;	// ɾ�˾�û��
	delete[] knownPoints;
	delete[] unknownPoints;
}
