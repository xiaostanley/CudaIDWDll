// 2016-5-24 8:32:29
// Stanley Xiao
// �������

#include "cudaIDWCPU.h"

#pragma warning (disable: 4819)

// �������

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

// CPU-IDW��ֵ
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
	float dfNominator = 0.0f;		// ����
	float dfDenominator = 0.0f;		// ��ĸ

	if (useManhattenDistance)	// ����Manhatten Distance
	{
		for (int i = 0; i < sizeOfUnknownPoints; i++)
		{
			dfNominator = 0.0f;
			dfDenominator = 0.0f;

			int j;
			for (j = 0; j < sizeOfKnownPoints; j++)
			{
				float weight = 0;
				float3 p = knownPoints[j];	// ��ǰ��֪��j

				// ���㵱ǰ��֪��j�뵱ǰδ֪��i֮��ľ���
				float d = calculateManhattenDistance(p.x, p.y, unknownPoints[i].x, unknownPoints[i].y);
				if (d < 0.000001f)	// Avoid division by 0 
				{
					break;
				}
				else
				{
					weight = 1.0f / powf(d, power);
					dfNominator += weight * p.z;		// ����=(Ȩ��*�߳�)֮��
					dfDenominator += weight;			// ��ĸ=Ȩ��֮��
				}
			}//end for

			// ����ǰδ֪��i��ĳһ��֪����뼫С����ֱ��ȡ����֪��ĸ߳�
			if (j != sizeOfKnownPoints)
				unknownPoints[i].z = knownPoints[j].z;
			else
				unknownPoints[i].z = dfNominator / dfDenominator;
		}
	}
	else	// ����Euclidean Distance
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
					float3 p = knownPoints[j];	// ��ǰ��֪��j

					float d = calculateEuclideanDistanceSqr(p.x, p.y, unknownPoints[i].x, unknownPoints[i].y);
					if (d < 0.000001f)
						break;
					else
					{
						weight = 1.0f / d;	// weight = 1 / pow(sqrt(XXX), 2.0f) = 1 / XXX
						dfNominator += weight * p.z;		// ����=(Ȩ��*�߳�)֮��
						dfDenominator += weight;			// ��ĸ=Ȩ��֮��
					}
				}//end for

				// ����ǰδ֪��i��ĳһ��֪����뼫С����ֱ��ȡ����֪��ĸ߳�
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
					float3 p = knownPoints[j];	// ��ǰ��֪��j

					float d = calculateEuclideanDistanceSqr(p.x, p.y, unknownPoints[i].x, unknownPoints[i].y);
					if (d < 0.000001f)
						break;
					else
					{
						weight = 1.0f / powf(d, (power / 2.0f));
						dfNominator += weight * p.z;		// ����=(Ȩ��*�߳�)֮��
						dfDenominator += weight;			// ��ĸ=Ȩ��֮��
					}
				}//end for

				// ����ǰδ֪��i��ĳһ��֪����뼫С����ֱ��ȡ����֪��ĸ߳�
				if (j != sizeOfKnownPoints)
					unknownPoints[i].z = knownPoints[j].z;
				else
					unknownPoints[i].z = dfNominator / dfDenominator;
			}
		}
	}
}
