// 2016-5-16 21:20:26
// Stanley Xiao
// ʹ��CUDA����IDW��ֵ

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
	);

#endif