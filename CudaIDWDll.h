// 2016-5-24 19:16:57
// Stanley Xiao
// CUDA IDW Interpolation

#ifndef _CUDA_IDW_DLL_H_
#define _CUDA_IDW_DLL_H_

// 下列 ifdef 块是创建使从 DLL 导出更简单的
// 宏的标准方法。此 DLL 中的所有文件都是用命令行上定义的 CUDAIDWDLL_EXPORTS
// 符号编译的。在使用此 DLL 的
// 任何其他项目上不应定义此符号。这样，源文件中包含此文件的任何其他项目都会将
// CUDAIDWDLL_API 函数视为是从 DLL 导入的，而此 DLL 则将用此宏定义的
// 符号视为是被导出的。
#ifdef CUDAIDWDLL_EXPORTS
#define CUDAIDWDLL_API __declspec(dllexport)
#else
#define CUDAIDWDLL_API __declspec(dllimport)
#endif

#include "cudaIDWH.h"

// 此类是从 CudaIDWDll.dll 导出的
class CUDAIDWDLL_API CudaIDWInterpolation 
{
public:
	CudaIDWInterpolation(void);
	
public:
	enum DistanceMode
	{
		EUCLIDEANDISTANCE,	// 欧氏距离
		MANHATTENDISTANCE	// 曼哈顿距离
	};

public:

	// 设置输入数据
	void setKnownPoints(GeoPoint3D* pData, int sizeOfData);

	// 设置DEM尺寸
	void setOriginalSize(int width, int height);

	// 是否采用GPU插值(默认启用)
	void setCudaDriven(bool enable);

	// 设置距离计算模式(默认是Euclidean Distance)
	void setDistanceMode(DistanceMode mode);

	// 设置待插值点
	void setUnknownPoints(float* x, float* y, int sizeOfData);

	// 执行IDW插值
	void execute(float power);
	// 执行IDW插值(默认反距离权为2.0)
	void execute(void);

	// 返回结果
	GeoPoint3D* getExpandedPoints(void);
	int getSizeOfExpandedPoints(void);

private:
	GeoPoint3D* sampledData;			// DEM经过采样后得到的点集
	int sizeOfSampledData;				// 采样点集的点数
	int width;							// DEM宽度
	int height;							// DEM高度
	float* unknownX;					// 待插值点X坐标
	float* unknownY;					// 待插值点Y坐标
	int sizeOfPointsForInterpolation;	// 待插值点点数
	bool isCudaDriven;					// 是否采用CUDA-IDW
	GeoPoint3D* expandedData;			// 插值后的点集
	int sizeOfExpandedData;				// 插值后的点集的点数
	DistanceMode mode;					// 距离计算方法
	float power;						// 反距离权
};

extern CUDAIDWDLL_API int nCudaIDWDll;

CUDAIDWDLL_API int fnCudaIDWDll(void);

#endif
