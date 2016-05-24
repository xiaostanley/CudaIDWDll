// 2016-5-16 21:20:26
// Stanley Xiao
// 使用CUDA进行IDW插值

#ifndef _CUDA_IDW_HEADER_H_
#define _CUDA_IDW_HEADER_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "device_functions_decls.h"

#include "GeoHelper.h"

#include <iostream>
#include <iomanip>
#include <time.h>

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
	);

#endif