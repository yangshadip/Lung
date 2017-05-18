#pragma once
#pragma omp for
#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
#include<memory>
#include<vector>
#include"LD.h"

#include "dcmtk\dcmdata\dctk.h"
#include "dcmtk\config\osconfig.h"
#include "DCMTK\dcmimgle\dcmimage.h"
#include "opencv2\opencv.hpp"
#include "dicomread.h"

#pragma comment (lib,"netapi32.lib")
#pragma comment (lib, "ofstd.lib")
#pragma comment (lib,"oflog.lib")
#pragma comment (lib, "dcmdata.lib")
#pragma comment (lib, "dcmimgle.lib")
#pragma comment (lib,"wsock32.lib")
#pragma comment (lib, "dcmimage.lib")
#pragma comment (lib, "opencv_world320.lib")

using namespace std;

int main()
{
	//Timer
	clock_t start, end;
	/********************* Model Parametre ********************** 
	*
	* @alpha mass energy
	* @beta bend energy
	* @lamda attraction energy
	* @delta gradient energy
	* @theta pot energy
	*/
	
	double alpha = 10;
	double beta = 9;
	double lamda = 9;
	double delta = 1.5;
	double theta = 1;

	/********************* Get Some Pointer **********************
	*
	* @pImg_short    Pointer to int16 Image
	* @pImg_double   Pointer to double Image
	* @pGradient     Pointer to Gradient Image
	* @pRegionGrow   Pointer to RegionGrow Image
	* @pMorphology   Pointer to Morphology Image
	* @pA            Pointer to Difference Image
	* @SliceNum      SliceNum After Interp
	* @Row Col Slice Coordinate Of Find Point
	*/

	start = clock();

	string folder = "C:\\Users\\yzw\\Desktop\\data\\LIDC-IDRI-0001";
	vector<string> files;
	FindDicomFile(folder + "\\*.dcm", files);

	//SliceNum After Interp
	int32_t SliceNum = 2*files.size() - 1;
	//*********************************

	shared_ptr<int16_t> pImg_short_smartpoint(GenerateVFromFolder(folder));
	shared_ptr<double> pImg_double_smartpoint(new double [MemorySize]);
	shared_ptr<double> pGradient_smartpoint(new double[MemorySize]);
	shared_ptr<double> pRegionGrow_smartpoint(new double[MemorySize]);
	shared_ptr<double> pMorphology_smartpoint(new double[MemorySize]);
	shared_ptr<double> pA_smartpoint(new double[MemorySize]);

	int16_t* pImg_short  = pImg_short_smartpoint.get();
	double* pImg_double  = pImg_double_smartpoint.get();
	double* pGradient    = pGradient_smartpoint.get();
	double* pRegionGrow  = pRegionGrow_smartpoint.get();
	double* pMorphology  = pMorphology_smartpoint.get();
	double* pA           = pA_smartpoint.get();
	
	//Get Pointer to Double Image
	for (int64_t i = 0; i < MemorySize; i++)
		*(pImg_double + i) = *(pImg_short + i);

	//Get Pointer to Gradient Image
	Gradient_3D(pImg_double, pGradient, SliceNum);

	//Get Pointer to RegionGrow Image
	Point RegionGrowSeedPoint = { 250, 150, floor(SliceNum / 2) };
	RegionGrow_3D(pImg_double, pRegionGrow, SliceNum, RegionGrowSeedPoint, 500);

	//Get Point to Difference Image
	Morphology(pRegionGrow, pMorphology, SliceNum, 10);

	for (int64_t i = 0; i < MemorySize; i++)
		*(pA + i) = ( *(pMorphology + i) - *(pRegionGrow + i) ) * (*(pImg_double + i));

	cv::Mat img = cv::Mat(RowSize, ColSize, CV_64FC1, pA+ RowSize*ColSize * 100, 0);
	imshow("test", img);
	cv::waitKey(100);
	//Get Candidate Points
	vector<int32_t> Row;
	vector<int32_t> Col;
	vector<int32_t> Slice;
	FindLocalMax(Row, Col, Slice, pA, SliceNum, 8);
	cout << "Find " << Row.size() << " Points" << endl;

	for (int64_t i = 0; i < Row.size(); i++)
		cout << "Point " << Row[i] << " " << Col[i] << " " << Slice[i] << endl;

	end = clock();
	cout << "Time: " << end - start << " ms" << endl;

	////Get Features
	//vector<double> surface;
	//vector<double> volume;
	//vector<double> sphericity;
	//vector<double> mean;
	//vector<double> stdn;
	//vector<double> skewness;
	//vector<double> kurtosis;

	//if (Row.size() == Col.size())
	//{
	//	cout << "Start!" << endl;
	//	int32_t N = Row.size();
	//	for (int32_t i = 0; i < N; i++)
	//	{
	//		Point sph_P = { Row[i],Col[i],Slice[i] };
	//		Sphere sph_co(sph_P);
	//		MinEnergy(sph_co, alpha, beta, lamda, delta, theta, pImg_double, pGradient);
	//		vector<double> value = GetValue(sph_co, pImg_double, NO);

	//		double constnum = sph_co.NumPerSlice*sph_co.SliceNum;

	//		double surface = calc_surf(sph_co);
	//		double volume = calc_vol(sph_co);
	//		double sphericity = Func_sphericity(volume, surface);
	//		double mean = Func_mean(value);
	//		double stdn = Func_std(value, mean);
	//		double skewness = Func_skewness(value, mean, stdn, constnum);
	//		double kurtosis = Func_kurtosis(value, mean, stdn, constnum);

	//		cout << "surface: " << surface << endl;
	//		cout << "volume: " << volume << endl;
	//		cout << "sphericity: " << sphericity << endl;
	//		cout << "mean: " << mean << endl;
	//		cout << "std: " << stdn << endl;
	//		cout << "skewness: " << skewness << endl;
	//		cout << "kurtosis: " << kurtosis << endl;

	//	}
	//}


	
	//cv::Mat img = cv::Mat(RowSize, ColSize, CV_64FC1, pImg_double + RowSize*ColSize * 100, 0);
	//imshow("test", img);
	//cv::waitKey(100);
	
	getchar();
	return 0;
}