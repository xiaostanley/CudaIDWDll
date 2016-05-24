// 2016-5-16 21:31:46
// Stanley Xiao
// cudaIDW的实现

#include "cudaIDWH.h"
#include "cudaIDWCPU.h"
#include "cudaIDWGPU.h"

// 是否使用GPU Shared Memory
#define _USE_GPU_SHARED_MEMORY_ 0

#pragma warning (disable: 4819)

/*
 * P.S. 
 * 1/ 约束关系(int* constraints)不在此处计算
 * 2/ IDW计算只针对新增点
 */

void cudaIDW(
	GeoPoint3D* sampledData,			// DEM经过采样后得到的点集
	int sizeOfSampledData,				// 采样点集的点数
	int width,							// DEM宽度
	int height,							// DEM高度
	float* unknownX,					// 待插值点X坐标
	float* unknownY,					// 待插值点Y坐标
	int sizeOfPointsForInterpolation,	// 待插值点点数
	bool isCudaDriven,					// 是否采用CUDA-IDW
	bool useManhanttenDistance,			// 是否采用Manhatten距离
	float power,						// 反距离权
	GeoPoint3D** expandedData,			// 插值后的点集
	int& sizeOfExpandedData				// 插值后的点集的点数
	)
{
	// 采用CUDA自定义结构体float3代替GeoPoint3D
	float3* knownPoints = new float3[sizeOfSampledData];
	for (int i = 0; i < sizeOfSampledData; i++)
		knownPoints[i] = make_float3(sampledData[i].x, sampledData[i].y, sampledData[i].z);

	// 最终输出结果的大小
	sizeOfExpandedData = sizeOfSampledData + sizeOfPointsForInterpolation;

	// 存储CPU-IDW算法的插值结果
	float3* unknownPoints = new float3[sizeOfPointsForInterpolation];
	// 初始化
	for (int i = 0; i < sizeOfPointsForInterpolation; i++)
		unknownPoints[i] = make_float3(unknownX[i], unknownY[i], -32767.0f);

	// debug 计时
	clock_t time0 = clock();

	if (isCudaDriven)
	{
		// 使用CUDA-IDW
#if _USE_GPU_SHARED_MEMORY_
		// 使用GPU共享内存
		IDWInGPUWithSM(unknownPoints, sizeOfPointsForInterpolation, knownPoints, sizeOfSampledData, power, useManhanttenDistance);
#else
		// 不使用GPU共享内存
		IDWInGPU(unknownPoints, sizeOfPointsForInterpolation, knownPoints, sizeOfSampledData, power, useManhanttenDistance);
#endif
	}
	else
	{
		// 使用CPU-IDW
		IDWInCPU(unknownPoints, sizeOfPointsForInterpolation, knownPoints, sizeOfSampledData, power, useManhanttenDistance);
	}

	// debug 计时
	std::cout << "IDW time = " << (double)(clock() - time0) / CLOCKS_PER_SEC << "s" << std::endl;

	// 存储插值结果
	GeoPoint3D* results = new GeoPoint3D[sizeOfExpandedData];
	for (int i = 0; i < sizeOfSampledData; i++)
		results[i] = GeoPoint3D(knownPoints[i].x, knownPoints[i].y, knownPoints[i].z);
	for (int i = sizeOfSampledData, j = 0; i < sizeOfExpandedData; i++, j++)
		results[i] = GeoPoint3D(unknownPoints[j].x, unknownPoints[j].y, unknownPoints[j].z);
	*expandedData = results;

	//delete[] results;	// 删了就没啦
	delete[] knownPoints;
	delete[] unknownPoints;
}
