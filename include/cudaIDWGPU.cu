// 2016-5-24 9:32:47
// Stanley Xiao
// 具体操作

#include "cudaIDWGPU.h"

#pragma warning (disable: 4819)

const int tileSize = 32;	// 线程块尺寸
const int tileSize2d = tileSize * tileSize;	// 线程块大小

#define THREAD_POS_IN_BLOCK		make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
#define SHARED_MEM_SIZE			49152
#define INVALID_POINT			make_float3(-32767.0f, -32767.0f, -32767.0f)

////////////////////////////////////////////////////////////////////

// 距离计算

// Euclidean Distance
__device__ __forceinline__ float calculateEuclideanDistanceSqrInGPU(
	float x1, 
	float y1, 
	float x2, 
	float y2
	)
{
	return ((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2));
}

// Manhatten Distance
__device__ __forceinline__ float calculateManhattenDistanceInGPU(
	float x1, 
	float y1, 
	float x2, 
	float y2
	)
{
	return (fabsf(x1 - x2) + fabsf(y1 - y2));
}

///////////////////////////////////////////////////////////////////

// power = 2.0
__global__ void interpolateSMEucDistPowerOf2(
	float3 *unknownPointsGpu,
	int sizeOfUnknownPoints,
	float3 const* __restrict__ knownPointsGpu,
	int sizeOfKnownPoints
	)
{
	const int2 threadPositionInBlock = THREAD_POS_IN_BLOCK;
	const int threadPositionInGrid = threadPositionInBlock.y * (blockDim.x * gridDim.x) + threadPositionInBlock.x;

	// 分配Shared Memory空间(per block)
	__shared__ float3 knownPointsInSM[tileSize2d];

	float nominator = 0.0f;
	float denominator = 0.0f;
	float singular = 0.0f;			//singular case where d(x, y) == 0;
	float singularValue = -1.0f;

	// 逐批地将已知点数据复制到SM中，以逐批计算分子分母
	for (int i = 0, tile = 0; i < sizeOfKnownPoints; i += tileSize2d, ++tile)
	{
		// 向Shared Memory中传输已知点数据
		if (threadPositionInGrid < sizeOfKnownPoints)
			knownPointsInSM[threadIdx.y * blockDim.x + threadIdx.x] = knownPointsGpu[tile * tileSize2d + (threadIdx.y * blockDim.x + threadIdx.x)];
		else
			knownPointsInSM[threadIdx.y * blockDim.x + threadIdx.x] = INVALID_POINT;
		__syncthreads();

		// 插值
		if (threadPositionInGrid < sizeOfUnknownPoints)
		{
			for (int i = 0; i < tileSize2d; i++)
			{
				float3 p = knownPointsInSM[i];
				if ((p.x == -32767.0f) && (p.y == -32767.0f) && (p.z == -32767.0f))
					break;

				float w = 0;
				float d = calculateEuclideanDistanceSqrInGPU(p.x, p.y, unknownPointsGpu[threadPositionInGrid].x, unknownPointsGpu[threadPositionInGrid].y);
				if (d < 0.000001f)
				{
					if (!singular)
					{
						singular = 1;
						singularValue = p.z;
					}
					break;
				}
				else
				{
					w = __fdividef(1.0f, d);
					//nominator += w * p.z;
					nominator = fmaf(w, p.z, nominator);
					denominator += w;
				}
			}
		}
		__syncthreads();
	}

	// 超出范围的点直接退出
	if (threadPositionInGrid >= sizeOfUnknownPoints)
		return;

	if (singular)
		unknownPointsGpu[threadPositionInGrid].z = singularValue;
	else
		unknownPointsGpu[threadPositionInGrid].z = nominator / denominator;
}

__global__ void interpolateSMEucDist(
	float3 *unknownPointsGpu,
	int sizeOfUnknownPoints,
	float3 const* __restrict__ knownPointsGpu,
	int sizeOfKnownPoints,
	float power
	)
{
	const int2 threadPositionInBlock = THREAD_POS_IN_BLOCK;
	const int threadPositionInGrid = threadPositionInBlock.y * (blockDim.x * gridDim.x) + threadPositionInBlock.x;

	// 分配Shared Memory空间(per block)
	__shared__ float3 knownPointsInSM[tileSize2d];

	float nominator = 0.0f;
	float denominator = 0.0f;
	float singular = 0.0f;			//singular case where d(x, y) == 0;
	float singularValue = -1.0f;

	// 逐批地将已知点数据复制到SM中，以逐批计算分子分母
	for (int i = 0, tile = 0; i < sizeOfKnownPoints; i += tileSize2d, ++tile)
	{
		// 向Shared Memory中传输已知点数据
		if (threadPositionInGrid < sizeOfKnownPoints)
			knownPointsInSM[threadIdx.y * blockDim.x + threadIdx.x] = knownPointsGpu[tile * tileSize2d + (threadIdx.y * blockDim.x + threadIdx.x)];
		else
			knownPointsInSM[threadIdx.y * blockDim.x + threadIdx.x] = INVALID_POINT;
		__syncthreads();

		// 插值
		if (threadPositionInGrid < sizeOfUnknownPoints)
		{
			for (int i = 0; i < tileSize2d; i++)
			{
				float3 p = knownPointsInSM[i];	// 已知点
				// 当前点为无效点(则其后所有点必为无效点)，则退出循环
				if ((p.x == -32767.0f) && (p.y == -32767.0f) && (p.z == -32767.0f))
					break;

				float w = 0;
				float d = calculateEuclideanDistanceSqrInGPU(p.x, p.y, unknownPointsGpu[threadPositionInGrid].x, unknownPointsGpu[threadPositionInGrid].y);

				if (d < 0.000001f)	// Avoid division by 0
				{
					if (!singular)
					{
						singular = 1;
						singularValue = p.z;
					}
					break;
				}
				else
				{
					w = __powf(d, -(power / 2.0f));
					//nominator += w * p.z;
					nominator = fmaf(w, p.z, nominator);
					denominator += w;
				}
			}
		}
		__syncthreads();
	}

	// 超出范围的点直接退出
	if (threadPositionInGrid >= sizeOfUnknownPoints)
		return;

	if (singular)
		unknownPointsGpu[threadPositionInGrid].z = singularValue;
	else
		unknownPointsGpu[threadPositionInGrid].z = nominator / denominator;
}


__global__ void interpolateSMManDist(
	float3 *unknownPointsGpu,
	int sizeOfUnknownPoints,
	float3 const* __restrict__ knownPointsGpu,
	int sizeOfKnownPoints,
	float power
	)
{
	const int2 threadPositionInBlock = THREAD_POS_IN_BLOCK;
	const int threadPositionInGrid = threadPositionInBlock.y * (blockDim.x * gridDim.x) + threadPositionInBlock.x;
	__shared__ float3 knownPointsInSM[tileSize2d];

	float nominator = 0.0f;
	float denominator = 0.0f;
	float singular = 0.0f;			//singular case where d(x, y) == 0;
	float singularValue = -1.0f;

	for (int i = 0, tile = 0; i < sizeOfKnownPoints; i += tileSize2d, ++tile)
	{
		if (threadPositionInGrid < sizeOfKnownPoints)
			knownPointsInSM[threadIdx.y * blockDim.x + threadIdx.x] = knownPointsGpu[tile * tileSize2d + (threadIdx.y * blockDim.x + threadIdx.x)];
		else
			knownPointsInSM[threadIdx.y * blockDim.x + threadIdx.x] = INVALID_POINT;
		__syncthreads();

		if (threadPositionInGrid < sizeOfUnknownPoints)
		{
			for (int i = 0; i < tileSize2d; i++)
			{
				float3 p = knownPointsInSM[i];
				if ((p.x == -32767.0f) && (p.y == -32767.0f) && (p.z == -32767.0f))
					break;

				float w = 0;
				float d = calculateManhattenDistanceInGPU(p.x, p.y, unknownPointsGpu[threadPositionInGrid].x, unknownPointsGpu[threadPositionInGrid].y);
				if (d < 0.000001f)	// Avoid division by 0
				{
					if (!singular)
					{
						singular = 1;
						singularValue = p.z;
					}
					break;
				}
				else
				{
					w = __powf(d, -power);
					//nominator += w * p.z;
					nominator = fmaf(w, p.z, nominator);
					denominator += w;
				}
			}
		}
		__syncthreads();
	}

	if (threadPositionInGrid >= sizeOfUnknownPoints)
		return;
	if (singular)	unknownPointsGpu[threadPositionInGrid].z = singularValue;
	else	unknownPointsGpu[threadPositionInGrid].z = nominator / denominator;
}


//////////////////////////////////////////////////////////////////////////////////

// 无Shared Memory的IDW插值

// GPU-IDW 核函数

__global__ void interpolateManDist(
	float3 *unknownPointsGpu,
	int sizeOfUnknownPoints,
	float3 const* __restrict__ knownPointsGpu,
	int sizeOfKnownPoints,
	float power
	)
{
	// 获取线程索引
	const int2 threadPositionInBlock = THREAD_POS_IN_BLOCK;
	const int threadPositionInGrid = threadPositionInBlock.y * (blockDim.x * gridDim.x) + threadPositionInBlock.x;

	// 超出范围的点不进行计算
	if (threadPositionInGrid >= sizeOfUnknownPoints)
		return;

	// 插值
	float nominator = 0.0f;	// 分子
	float denominator = 0.0f;	// 分母

	int i = 0;
	for (i = 0; i < sizeOfKnownPoints; i++)
	{
		float weight = 0;
		float3 p = knownPointsGpu[i];

		float d = calculateManhattenDistanceInGPU(p.x, p.y, unknownPointsGpu[threadPositionInGrid].x, unknownPointsGpu[threadPositionInGrid].y);
		if (d < 0.000001f)
			break;
		else
		{
			weight = __powf(d, -power);
			nominator = fmaf(weight, p.z, nominator);
			denominator += weight;
		}
	}

	if (i != sizeOfKnownPoints)
		unknownPointsGpu[threadPositionInGrid].z = knownPointsGpu[i].z;
	else
		unknownPointsGpu[threadPositionInGrid].z = __fdividef(nominator, denominator);
}

// power = 2.0f，则权重 w = 1/ d
__global__ void interpolateEucDistPowerOf2(
	float3* unknownPointsGpu,
	int sizeOfUnknownPoints,
	float3 const* __restrict__ knownPointsGpu,
	int sizeOfKnownPoints
	)
{
	const int2 threadPositionInBlock = THREAD_POS_IN_BLOCK;
	const int threadPositionInGrid = threadPositionInBlock.y * (blockDim.x * gridDim.x) + threadPositionInBlock.x;
	if (threadPositionInGrid >= sizeOfUnknownPoints)
		return;
	
	float nominator = 0.0f;	// 分子
	float denominator = 0.0f;	// 分母

	int i = 0;
	for (i = 0; i < sizeOfKnownPoints; i++)
	{
		float weight = 0;
		float3 p = knownPointsGpu[i];

		float d = calculateEuclideanDistanceSqrInGPU(p.x, p.y, unknownPointsGpu[threadPositionInGrid].x, unknownPointsGpu[threadPositionInGrid].y);
		if (d < 0.000001f)
			break;
		else
		{
			weight = __fdividef(1.0f, d);
			nominator = fmaf(weight, p.z, nominator);
			denominator += weight;
		}
	}

	if (i != sizeOfKnownPoints)
		unknownPointsGpu[threadPositionInGrid].z = knownPointsGpu[i].z;
	else
		unknownPointsGpu[threadPositionInGrid].z = __fdividef(nominator, denominator);
}

// power > 2.0f, 则权重 w = 1 / power(d, power/2)
__global__ void interpolateEucDist(
	float3* unknownPointsGpu,
	int sizeOfUnknownPoints,
	float3 const* __restrict__ knownPointsGpu,
	int sizeOfKnownPoints,
	float power
	)
{
	const int2 threadPositionInBlock = THREAD_POS_IN_BLOCK;
	const int threadPositionInGrid = threadPositionInBlock.y * (blockDim.x * gridDim.x) + threadPositionInBlock.x;
	if (threadPositionInGrid >= sizeOfUnknownPoints)
		return;

	float nominator = 0.0f;	// 分子
	float denominator = 0.0f;	// 分母

	int i = 0;
	for (i = 0; i < sizeOfKnownPoints; i++)
	{
		float weight = 0;
		float3 p = knownPointsGpu[i];

		float d = calculateEuclideanDistanceSqrInGPU(p.x, p.y, unknownPointsGpu[threadPositionInGrid].x, unknownPointsGpu[threadPositionInGrid].y);
		if (d < 0.000001f)
			break;
		else
		{
			weight = __powf(d, -(power / 2.0f));
			nominator = fmaf(weight, p.z, nominator);
			denominator += weight;
		}
	}

	if (i != sizeOfKnownPoints)
		unknownPointsGpu[threadPositionInGrid].z = knownPointsGpu[i].z;
	else
		unknownPointsGpu[threadPositionInGrid].z = __fdividef(nominator, denominator);
}

///////////////////////////////////////////////////////////////////////////////////

// GPU-IDW 不使用Shared Memory
extern "C"
void IDWInGPU(
	float3* unknownPoints,
	int sizeOfUnknownPoints,
	float3* knownPoints,
	int sizeOfKnownPoints,
	float power,
	bool useManhattenDistance
)
{
	// debug 计时
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// 在GPU上分配已知点和未知点的存储空间
	float3* knownPointsGpu;
	cudaMalloc((void**)&knownPointsGpu, sizeof(float3) * sizeOfKnownPoints);
	cudaMemcpy(knownPointsGpu, knownPoints, sizeof(float3) * sizeOfKnownPoints, cudaMemcpyHostToDevice);
	float3* unknownPointsGpu;
	cudaMalloc((void**)&unknownPointsGpu, sizeof(float3) * sizeOfUnknownPoints);
	cudaMemcpy(unknownPointsGpu, unknownPoints, sizeof(float3) * sizeOfUnknownPoints, cudaMemcpyHostToDevice);

	// 每个block由32*32个线程组成
	// 每个grid由 w/32 * h/32 个block组成
	dim3 block = dim3(tileSize, tileSize, 1);
	dim3 grid;
	int countOfBlocks = sizeOfUnknownPoints / tileSize2d + ((sizeOfUnknownPoints % tileSize2d == 0) ? 0 : 1);
	if (countOfBlocks >= 65536)
		grid = dim3(65536, (sizeOfUnknownPoints / tileSize2d), 1);
	else
		grid = dim3(countOfBlocks, 1, 1);

	//------------------
	// --USING GLOBAL MEMORY
	//---------------------
	cudaDeviceSetCacheConfig(cudaFuncCache::cudaFuncCachePreferL1);

	// debug 计时
	cudaEventRecord(start);

	// 执行插值
	if (useManhattenDistance)
		interpolateManDist << <grid, block >> >(unknownPointsGpu, sizeOfUnknownPoints, knownPointsGpu, sizeOfKnownPoints, power);
	else if (power == 2.0f)
		interpolateEucDistPowerOf2 << <grid, block >> >(unknownPointsGpu, sizeOfUnknownPoints, knownPointsGpu, sizeOfKnownPoints);
	else
		interpolateEucDist << <grid, block >> > (unknownPointsGpu, sizeOfUnknownPoints, knownPointsGpu, sizeOfKnownPoints, power);

	// 返回插值结果
	cudaMemcpy(unknownPoints, unknownPointsGpu, sizeof(float3) * sizeOfUnknownPoints, cudaMemcpyDeviceToHost);

	// debug 计时
	cudaEventRecord(stop);
	cudaDeviceSynchronize();
	float msecTotal;
	cudaEventElapsedTime(&msecTotal, start, stop);
	std::cout << "msecTimeCUDA time = " << msecTotal << "ms" << std::endl;

	cudaFree(knownPointsGpu);
	cudaFree(unknownPointsGpu);
}

// GPU-IDW 使用Shared Memory
void IDWInGPUWithSM(
	float3* unknownPoints,
	int sizeOfUnknownPoints,
	float3* knownPoints,
	int sizeOfKnownPoints,
	float power,
	bool useManhattenDistance
	)
{
	// debug 计时
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// 设置线程
	dim3 block = dim3(tileSize, tileSize, 1);
	dim3 grid;
	int countOfBlocks = sizeOfUnknownPoints / tileSize2d + ((sizeOfUnknownPoints % tileSize2d == 0) ? 0 : 1);
	if (countOfBlocks >= 65536)
		grid = dim3(65536, (sizeOfUnknownPoints / tileSize2d), 1);
	else
		grid = dim3(countOfBlocks, 1, 1);

	// 扩充已知点数组
	int invalidSizeOfExpandedKnownPoints = sizeOfKnownPoints % tileSize2d;
	int sizeOfExpandedKnownPoints = (sizeOfKnownPoints / tileSize2d + ((invalidSizeOfExpandedKnownPoints == 0) ? 0 : 1))*tileSize2d;

	// 在GPU上分配已知点存储空间
	float3* knownPointsGpu;
	cudaMalloc((void**)&knownPointsGpu, sizeof(float3) * sizeOfExpandedKnownPoints);
	cudaMemcpy(knownPointsGpu, knownPoints, sizeof(float3) * sizeOfKnownPoints, cudaMemcpyHostToDevice);
	if (invalidSizeOfExpandedKnownPoints != 0)
	{
		float3* knownPointsForExpansion = new float3[invalidSizeOfExpandedKnownPoints];
		for (int i = 0; i < invalidSizeOfExpandedKnownPoints; i++)
			knownPointsForExpansion[i] = INVALID_POINT;
		cudaMemcpy(knownPointsGpu + sizeOfKnownPoints, knownPointsForExpansion, sizeof(float3) * invalidSizeOfExpandedKnownPoints, cudaMemcpyHostToDevice);
	}

	// 在GPU上分配未知点存储空间
	float3* unknownPointsGpu;
	cudaMalloc((void**)&unknownPointsGpu, sizeof(float3) * sizeOfUnknownPoints);
	cudaMemcpy(unknownPointsGpu, unknownPoints, sizeof(float3) * sizeOfUnknownPoints, cudaMemcpyHostToDevice);

	//---------------------
	// --USING SHARED MEMORY
	//---------------------
	cudaDeviceSetCacheConfig(cudaFuncCache::cudaFuncCachePreferShared);

	// debug 计时
	cudaEventRecord(start);

	// 执行插值
	if (useManhattenDistance)
		interpolateSMManDist << <grid, block >> >(unknownPointsGpu, sizeOfUnknownPoints, knownPointsGpu, sizeOfKnownPoints, power);
	else if (power == 2.0f)
		interpolateEucDistPowerOf2 << <grid, block >> >(unknownPointsGpu, sizeOfUnknownPoints, knownPointsGpu, sizeOfKnownPoints);
	else
		interpolateSMEucDist << <grid, block >> >(unknownPointsGpu, sizeOfUnknownPoints, knownPointsGpu, sizeOfKnownPoints, power);

	// 返回插值结果
	cudaMemcpy(unknownPoints, unknownPointsGpu, sizeof(float3) * sizeOfUnknownPoints, cudaMemcpyDeviceToHost);

	// debug 计时
	cudaEventRecord(stop);
	cudaDeviceSynchronize();
	cudaGetLastError();
	float msecTotal;
	cudaEventElapsedTime(&msecTotal, start, stop);
	std::cout << "msecTimeCUDAShared time = " << msecTotal << "ms" << std::endl;

	cudaFree(knownPointsGpu);
	cudaFree(unknownPointsGpu);
}
