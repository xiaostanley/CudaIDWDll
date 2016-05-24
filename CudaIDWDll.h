// 2016-5-24 19:16:57
// Stanley Xiao
// CUDA IDW Interpolation

#ifndef _CUDA_IDW_DLL_H_
#define _CUDA_IDW_DLL_H_

// ���� ifdef ���Ǵ���ʹ�� DLL �������򵥵�
// ��ı�׼�������� DLL �е������ļ��������������϶���� CUDAIDWDLL_EXPORTS
// ���ű���ġ���ʹ�ô� DLL ��
// �κ�������Ŀ�ϲ�Ӧ����˷��š�������Դ�ļ��а������ļ����κ�������Ŀ���Ὣ
// CUDAIDWDLL_API ������Ϊ�Ǵ� DLL ����ģ����� DLL ���ô˺궨���
// ������Ϊ�Ǳ������ġ�
#ifdef CUDAIDWDLL_EXPORTS
#define CUDAIDWDLL_API __declspec(dllexport)
#else
#define CUDAIDWDLL_API __declspec(dllimport)
#endif

#include "cudaIDWH.h"

// �����Ǵ� CudaIDWDll.dll ������
class CUDAIDWDLL_API CudaIDWInterpolation 
{
public:
	CudaIDWInterpolation(void);
	
public:
	enum DistanceMode
	{
		EUCLIDEANDISTANCE,	// ŷ�Ͼ���
		MANHATTENDISTANCE	// �����پ���
	};

public:

	// ������������
	void setKnownPoints(GeoPoint3D* pData, int sizeOfData);

	// ����DEM�ߴ�
	void setOriginalSize(int width, int height);

	// �Ƿ����GPU��ֵ(Ĭ������)
	void setCudaDriven(bool enable);

	// ���þ������ģʽ(Ĭ����Euclidean Distance)
	void setDistanceMode(DistanceMode mode);

	// ���ô���ֵ��
	void setUnknownPoints(float* x, float* y, int sizeOfData);

	// ִ��IDW��ֵ
	void execute(float power);
	// ִ��IDW��ֵ(Ĭ�Ϸ�����ȨΪ2.0)
	void execute(void);

	// ���ؽ��
	GeoPoint3D* getExpandedPoints(void);
	int getSizeOfExpandedPoints(void);

private:
	GeoPoint3D* sampledData;			// DEM����������õ��ĵ㼯
	int sizeOfSampledData;				// �����㼯�ĵ���
	int width;							// DEM���
	int height;							// DEM�߶�
	float* unknownX;					// ����ֵ��X����
	float* unknownY;					// ����ֵ��Y����
	int sizeOfPointsForInterpolation;	// ����ֵ�����
	bool isCudaDriven;					// �Ƿ����CUDA-IDW
	GeoPoint3D* expandedData;			// ��ֵ��ĵ㼯
	int sizeOfExpandedData;				// ��ֵ��ĵ㼯�ĵ���
	DistanceMode mode;					// ������㷽��
	float power;						// ������Ȩ
};

extern CUDAIDWDLL_API int nCudaIDWDll;

CUDAIDWDLL_API int fnCudaIDWDll(void);

#endif
