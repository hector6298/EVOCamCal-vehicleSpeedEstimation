#pragma once
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#define IM1 2147483563
#define IM2 2147483399
#define AM (1.0/IM1)
#define IMM1 (IM1-1)
#define IA1 40014
#define IA2 40692
#define IQ1 53668
#define IQ2 52774
#define IR1 12211
#define IR2 3791
#define NTAB 32
#define NDIV (1+IMM1/NTAB)
#define EPS1 1.2e-7
#define RNMX (1.0-EPS1)


//! define the initial population of EDA (default: 20000)
#define EDA_INIT_POP (1000000)
//! define the selected population of EDA (default: 20)
#define EDA_SEL_POP (10000)
//! define the number of iterations of EDA (default: 100)
#define EDA_ITER_NUM (1000)
//! define the threshold of ratio of reprojection errors between iterations (default: 0.10)
#define EDA_REPROJ_ERR_THLD (0.0001)
//! defines range for 2D point variation 
#define EDA_RNG_2DPT (0.005)
//! defines range for 3D point variation
#define EDA_RNG_3DPT (0.005)

static double rand2(long idum)
{
	int j;
	long k;
	static long idum2 = 123456789;
	static long iy = 0;
	static long iv[NTAB];
	double temp;
    
	if (idum <= 0) {
		if (-(idum) < 1) idum = 1;
		else idum = -(idum);
		idum2 = (idum);
		for (j = NTAB + 7; j >= 0; j--) {
			k = (idum) / IQ1;
			idum = IA1*(idum - k*IQ1) - k*IR1;
			if (idum < 0) idum += IM1;
			if (j < NTAB) iv[j] = idum;
		}
		iy = iv[0];
	}
    
	k = (idum) / IQ1;
	idum = IA1*(idum - k*IQ1) - k*IR1;
	if (idum < 0) idum += IM1;
	k = idum2 / IQ2;
	idum2 = IA2*(idum2 - k*IQ2) - k*IR2;
	if (idum2 < 0) idum2 += IM2;
	j = iy / NDIV;
	iy = iv[j] - idum2;
	iv[j] = idum;
	if (iy < 1) iy += IMM1;
    
	if ((temp = AM*iy) > RNMX) return RNMX;
	else return temp;
}

//! generates a random numberv
static double get_rand_num(float max, float min, long seed)
{
	int rand = rand2(seed);
	int duration = max - min;
	return min + rand*duration;
}

static double haversine_distance(cv::Point2f pt3d1, cv::Point2f pt3d2){
    // distance between latitudes 
    // and longitudes 
    double lat1 = pt3d1.y, lat2 = pt3d2.y;
    double lon1 = pt3d1.x, lon2 = pt3d2.x;

    double dLat = (lat2 - lat1) * 
                M_PI / 180.0; 
    double dLon = (lon2 - lon1) *  
                M_PI / 180.0; 

    // convert to radians 
    lat1 = (lat1) * M_PI / 180.0; 
    lat2 = (lat2) * M_PI / 180.0; 

    // apply formulae 
    double a = pow(sin(dLat / 2), 2) +  
            pow(sin(dLon / 2), 2) *  
            cos(lat1) * cos(lat2); 
    double rad = 6371; 
    double c = 2 * asin(sqrt(a)); 
    return rad * c; 
} 

static std::vector<std::vector<cv::Point2f>> gen_pairs(std::vector<cv::Point2f> pts){
    std::vector<std::vector<cv::Point2f>> result;
    for(int i = 0; i < pts.size(); i++){
        for(int j = i+1; j < pts.size(); j++){
            std::vector<cv::Point2f> pair;
            pair.push_back(pts[i]);
            pair.push_back(pts[j]);
            result.push_back(pair);
        }
    }
    return result;
}

static cv::Point2f backproj2D3D(cv::Point2f pt2d, cv::Mat homoMat){
    cv::Mat homoPt2d(3,1, CV_64F);
    cv::Mat hh(3,1, CV_64F);
    homoPt2d.at<double>(0,0) = pt2d.x;
    homoPt2d.at<double>(1,0) = pt2d.y;
    homoPt2d.at<double>(2,0) = 1;

    cv::Mat invMat = homoMat.inv();
    hh = invMat * homoPt2d;
    
    cv::Point2f pt3d;
    pt3d.x =  (hh.at<double>(0,0) / hh.at<double>(2,0));
    pt3d.y =  (hh.at<double>(1,0) / hh.at<double>(2,0));

    return pt3d;
}

static cv::Point2f proj3D2D(cv::Point2f pt3d, cv::Mat homoMat){
    cv::Mat o3dPtMat(3, 1, CV_64F);
    cv::Mat o2dPtMat(3, 1, CV_64F);
    cv::Point2f o2dPt;

    o3dPtMat.at<double>(0, 0) = pt3d.x;
    o3dPtMat.at<double>(1, 0) = pt3d.y;
    o3dPtMat.at<double>(2, 0) = 1;
    o2dPtMat = homoMat * o3dPtMat;
    o2dPt = cv::Point2f((o2dPtMat.at<double>(0, 0) / o2dPtMat.at<double>(2, 0)), (o2dPtMat.at<double>(1, 0) / o2dPtMat.at<double>(2, 0)));

    return o2dPt;
}
