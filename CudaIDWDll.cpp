// CudaIDWDll.cpp : 定义 DLL 应用程序的导出函数。
//

#include "stdafx.h"
#include "CudaIDWDll.h"


// 这是导出变量的一个示例
CUDAIDWDLL_API int nCudaIDWDll=0;

// 这是导出函数的一个示例。
CUDAIDWDLL_API int fnCudaIDWDll(void)
{
	return 42;
}

// 这是已导出类的构造函数。
// 有关类定义的信息，请参阅 CudaIDWDll.h
CudaIDWInterpolation::CudaIDWInterpolation( void )
	: sampledData(NULL),
	  sizeOfSampledData(0),
	  width(0),
	  height(0),
	  isCudaDriven(true),
	  expandedData(NULL),
	  sizeOfExpandedData(0),
	  mode(EUCLIDEANDISTANCE),
	  power(2.0f)
{
}

void CudaIDWInterpolation::setKnownPoints( GeoPoint3D* pData, int sizeOfData )
{
	sampledData = pData;
	sizeOfSampledData = sizeOfData;
}

void CudaIDWInterpolation::setOriginalSize( int width, int height )
{
	this->width = width;
	this->height = height;
}

void CudaIDWInterpolation::setCudaDriven( bool enable )
{
	isCudaDriven = enable;
}

void CudaIDWInterpolation::setDistanceMode( DistanceMode mode )
{
	this->mode = mode;
}

void CudaIDWInterpolation::setUnknownPoints( float* x, float* y, int sizeOfData )
{
	unknownX = x;
	unknownY = y;
	sizeOfPointsForInterpolation = sizeOfData;
}

void CudaIDWInterpolation::execute( float power )
{
	cudaIDW(
		sampledData,
		sizeOfSampledData,
		width,
		height,
		unknownX,
		unknownY,
		sizeOfPointsForInterpolation,
		isCudaDriven,
		((mode == MANHATTENDISTANCE)? true : false),
		power,
		&expandedData,
		sizeOfExpandedData
		);
}

void CudaIDWInterpolation::execute( void )
{
	cudaIDW(
		sampledData,
		sizeOfSampledData,
		width,
		height,
		unknownX,
		unknownY,
		sizeOfPointsForInterpolation,
		isCudaDriven,
		((mode == MANHATTENDISTANCE)? true : false),
		2.0f,
		&expandedData,
		sizeOfExpandedData
		);
}

GeoPoint3D* CudaIDWInterpolation::getExpandedPoints( void )
{
	return expandedData;
}

int CudaIDWInterpolation::getSizeOfExpandedPoints( void )
{
	return sizeOfExpandedData;
}
