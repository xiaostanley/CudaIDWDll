// 2016-5-16 21:12:34
// Stanley Xiao
// Geo Data Processing Helper Functions

#ifndef _GEO_HELPER_H_
#define _GEO_HELPER_H_

struct GeoPoint3D
{
	GeoPoint3D(void) {}
	GeoPoint3D(float _x, float _y, float _z)
		:x(_x), y(_y), z(_z)
	{}

	float x;
	float y;
	float z;
};

#endif