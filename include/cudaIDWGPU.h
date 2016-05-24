// 2016-5-24 9:31:00
// Stanley Xiao
// GPU IDW����ʵ��

#ifndef _CUDA_IDW_GPU_H
#define _CUDA_IDW_GPU_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <device_functions_decls.h>

#include <iostream>

// GPU-IDW ��ʹ��Shared Memory
extern "C"
void IDWInGPU(
	float3* unknownPoints,
	int sizeOfUnknownPoints,
	float3* knownPoints,
	int sizeOfKnownPoints,
	float power,
	bool useManhattenDistance
	);

// GPU-IDW ʹ��Shared Memory
extern "C"
void IDWInGPUWithSM(
	float3* unknownPoints,
	int sizeOfUnknownPoints,
	float3* knownPoints,
	int sizeOfKnownPoints,
	float power,
	bool useManhattenDistance
	);

#endif
