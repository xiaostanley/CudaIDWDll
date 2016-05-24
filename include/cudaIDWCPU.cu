// 2016-5-24 8:32:29
// Stanley Xiao
// 具体操作

#include "cudaIDWCPU.h"

#pragma warning (disable: 4819)

// 距离计算

// Euclidean Distance
__host__ __forceinline__ float calculateEuclideanDistanceSqr(
	float x1, 
	float y1, 
	float x2, 
	float y2
	)
{
	return ((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2));
}

// Manhatten Distance
__host__ __forceinline__ float calculateManhattenDistance(
	float x1, 
	float y1, 
	float x2, 
	float y2
	)
{
	return (fabsf(x1 - x2) + fabsf(y1 - y2));
}

// CPU-IDW插值
extern "C"
void IDWInCPU(
	float3* unknownPoints,
	int sizeOfUnknownPoints,
	float3* knownPoints,
	int sizeOfKnownPoints,
	float power,
	bool useManhattenDistance
)
{
	float dfNominator = 0.0f;		// 分子
	float dfDenominator = 0.0f;		// 分母

	if (useManhattenDistance)	// 采用Manhatten Distance
	{
		for (int i = 0; i < sizeOfUnknownPoints; i++)
		{
			dfNominator = 0.0f;
			dfDenominator = 0.0f;

			int j;
			for (j = 0; j < sizeOfKnownPoints; j++)
			{
				float weight = 0;
				float3 p = knownPoints[j];	// 当前已知点j

				// 计算当前已知点j与当前未知点i之间的距离
				float d = calculateManhattenDistance(p.x, p.y, unknownPoints[i].x, unknownPoints[i].y);
				if (d < 0.000001f)	// Avoid division by 0 
				{
					break;
				}
				else
				{
					weight = 1.0f / powf(d, power);
					dfNominator += weight * p.z;		// 分子=(权重*高程)之和
					dfDenominator += weight;			// 分母=权重之和
				}
			}//end for

			// 若当前未知点i与某一已知点距离极小，则直接取该已知点的高程
			if (j != sizeOfKnownPoints)
				unknownPoints[i].z = knownPoints[j].z;
			else
				unknownPoints[i].z = dfNominator / dfDenominator;
		}
	}
	else	// 采用Euclidean Distance
	{
		if (power == 2.0f)
		{
			for (int i = 0; i < sizeOfUnknownPoints; i++)
			{
				dfNominator = 0.0f;
				dfDenominator = 0.0f;
				int j;
				for (j = 0; j < sizeOfKnownPoints; j++)
				{
					float weight = 0;
					float3 p = knownPoints[j];	// 当前已知点j

					float d = calculateEuclideanDistanceSqr(p.x, p.y, unknownPoints[i].x, unknownPoints[i].y);
					if (d < 0.000001f)
						break;
					else
					{
						weight = 1.0f / d;	// weight = 1 / pow(sqrt(XXX), 2.0f) = 1 / XXX
						dfNominator += weight * p.z;		// 分子=(权重*高程)之和
						dfDenominator += weight;			// 分母=权重之和
					}
				}//end for

				// 若当前未知点i与某一已知点距离极小，则直接取该已知点的高程
				if (j != sizeOfKnownPoints)
					unknownPoints[i].z = knownPoints[j].z;
				else
					unknownPoints[i].z = dfNominator / dfDenominator;
			}
		} 
		else
		{
			for (int i = 0; i < sizeOfUnknownPoints; i++)
			{
				dfNominator = 0.0f;
				dfDenominator = 0.0f;

				int j;
				for (j = 0; j < sizeOfKnownPoints; j++)
				{
					float weight = 0;
					float3 p = knownPoints[j];	// 当前已知点j

					float d = calculateEuclideanDistanceSqr(p.x, p.y, unknownPoints[i].x, unknownPoints[i].y);
					if (d < 0.000001f)
						break;
					else
					{
						weight = 1.0f / powf(d, (power / 2.0f));
						dfNominator += weight * p.z;		// 分子=(权重*高程)之和
						dfDenominator += weight;			// 分母=权重之和
					}
				}//end for

				// 若当前未知点i与某一已知点距离极小，则直接取该已知点的高程
				if (j != sizeOfKnownPoints)
					unknownPoints[i].z = knownPoints[j].z;
				else
					unknownPoints[i].z = dfNominator / dfDenominator;
			}
		}
	}
}
