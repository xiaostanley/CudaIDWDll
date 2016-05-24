// 2016-5-24 8:47:59
// Stanley Xiao
// CPU IDW 具体实现

#ifndef _CUDA_IDW_EXECUTE_H_
#define _CUDA_IDW_CPU_H_

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <device_functions_decls.h>

// CPU-IDW插值
extern "C"
void IDWInCPU(
	float3* unknownPoints,
	int sizeOfUnknownPoints,
	float3* knownPoints,
	int sizeOfKnownPoints,
	float power,
	bool useManhattenDistance
	);

#endif
