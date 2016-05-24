// CudaIDWDll.cpp : ���� DLL Ӧ�ó���ĵ���������
//

#include "stdafx.h"
#include "CudaIDWDll.h"


// ���ǵ���������һ��ʾ��
CUDAIDWDLL_API int nCudaIDWDll=0;

// ���ǵ���������һ��ʾ����
CUDAIDWDLL_API int fnCudaIDWDll(void)
{
	return 42;
}

// �����ѵ�����Ĺ��캯����
// �й��ඨ�����Ϣ������� CudaIDWDll.h
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
