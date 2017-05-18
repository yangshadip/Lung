#pragma once
#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
#include<memory>
#include<vector>
#include"LD.h"
#include<math.h>

#include "dcmtk\dcmdata\dctk.h"
#include "dcmtk\config\osconfig.h"
#include "DCMTK\dcmimgle\dcmimage.h"
#include "opencv2\opencv.hpp"

#define MemorySize 512 * 512 * SliceNum
#define RowSize 512
#define ColSize 512
#define value(p,r,c,s) *(p + RowSize*ColSize*(s - 1) + ColSize*(r - 1) + (c - 1) )

#define f_1(i1,j1,i2,j2) (	(sph_co.GetLoc(i1,j1).RowNum - sph_co.GetLoc(i2,j2).RowNum)*(sph_co.GetLoc(i1,j1).RowNum - sph_co.GetLoc(i2,j2).RowNum) + (sph_co.GetLoc(i1,j1).ColNum - sph_co.GetLoc(i2,j2).ColNum)*(sph_co.GetLoc(i1,j1).ColNum - sph_co.GetLoc(i2,j2).ColNum) + (sph_co.GetLoc(i1,j1).SliceNum - sph_co.GetLoc(i2,j2).SliceNum)*(sph_co.GetLoc(i1,j1).SliceNum - sph_co.GetLoc(i2,j2).SliceNum)	)
#define f_2(i1,j1,i2,j2,i3,j3) ( (sph_co.GetLoc(i1,j1).RowNum - 2*sph_co.GetLoc(i2,j2).RowNum + sph_co.GetLoc(i3,j3).RowNum)*(sph_co.GetLoc(i1,j1).RowNum - 2*sph_co.GetLoc(i2,j2).RowNum + sph_co.GetLoc(i3,j3).RowNum) + (sph_co.GetLoc(i1,j1).ColNum - 2*sph_co.GetLoc(i2,j2).ColNum + sph_co.GetLoc(i3,j3).ColNum)*(sph_co.GetLoc(i1,j1).ColNum - 2*sph_co.GetLoc(i2,j2).ColNum + sph_co.GetLoc(i3,j3).ColNum) +(sph_co.GetLoc(i1,j1).SliceNum - 2*sph_co.GetLoc(i2,j2).SliceNum + sph_co.GetLoc(i3,j3).SliceNum)*(sph_co.GetLoc(i1,j1).SliceNum - 2*sph_co.GetLoc(i2,j2).SliceNum + sph_co.GetLoc(i3,j3).SliceNum) )

double const PI = 3.14159265;


using namespace std;

enum IFSliceBetweenSlice 
{
	YES,
	NO 
};

struct Point
{
	int32_t RowNum;
	int32_t ColNum;
	int32_t SliceNum;
};


void Pol2Cart(double &X, double &Y, double Angle, double Radius);


struct Img
{
	short* p;
	int RowNum;
	int ColNum;
	int SliceNum;
};
class Voxel3D
{
public: 
	Voxel3D(shared_ptr<short> p,int SliceNum) :pVoxel(p),SliceNum(SliceNum){};
private: shared_ptr<short> pVoxel;
		 int SliceNum;
};

class MyStack
{
public:
	MyStack() 
	{
		MaxCount = 512 * 512 * 1000;
		_data = new Point [MaxCount];
		_t = -1;
	};
	~MyStack() { delete[] _data; }
	bool empty() { return _t < 0; }
	bool full() { return _t >= MaxCount; }
	void push(Point data)
	{
		if (full())
		{
			return;
		}
		_t++;
		*(_data + _t) = data;
	}

	bool pop(Point& t)
	{
		if (empty())
		{
			return false;
		}
		t = *(_data+_t);
		_t--;
		return true;
	}
	long long showtop() { return _t; }
private: Point* _data;
		 long long MaxCount;
		 long long _t;
};
//*************************************************************
class Sphere
{
public:
	Sphere(Point CentrePoint)
	{
		this->CentrePoint = CentrePoint;
		double Radius = 15;
		double SliceInternal = 1;
		this->NumPerSlice = 10;
		this->SliceNum = round((Radius * 2 - 1) / SliceInternal);
		_d = vector<vector<Point>>(SliceNum, vector<Point>(NumPerSlice));

		for (auto i = 0; i < SliceNum; i++)
		{
			double r = sqrt(Radius*Radius - ((i + 1)*SliceInternal - Radius)*((i + 1)*SliceInternal - Radius));
			for (auto j = 0; j < NumPerSlice; j++)
			{
				double angle = 2 * PI * j / NumPerSlice;
				double x = r*cos(angle);
				double y = r*sin(angle);
				/*Pol2Cart(x, y, 2 * PI * j / NumPerSlice, r);*/
				double RowTemp = -y + CentrePoint.RowNum;
				double ColTemp = x + CentrePoint.ColNum;
				double SliceTemp = (i + 1)*SliceInternal - Radius + CentrePoint.SliceNum;

				_d[i][j].RowNum = round(RowTemp);
				_d[i][j].ColNum = round(ColTemp);
				_d[i][j].SliceNum = round(SliceTemp);
			}
		}
		
	}
	~Sphere() {};


	Point GetLoc(int s, int n) { return _d[s - 1][n - 1]; }// s层 n个
	void SetLoc(int s, int n, Point point) { _d[s - 1][n - 1] = point; }
	int SliceNum;
	int NumPerSlice;
	Point CentrePoint;
    vector<vector<Point>> _d;

};

short* GenerateVFromFolder(string folder);//从一个文件夹创建三维图像
int FindDicomFile(string const path, vector<string>& files);//从路径寻找dicom文件放入vector files
void RegionGrow_3D(double* const input, double* const output, int32_t SliceNum, Point SeedPoint, int32_t Threshold);
void Gradient_3D(double* const input, double* const output, int32_t SliceNum);//三维区域生长
void Morphology(double* const input, double* const output, int32_t SliceNum, int32_t element_size);
void FindLocalMax(vector<int32_t>& Row, vector<int32_t>& Col, vector<int32_t>& Slice, double * const pA, int32_t SliceNum, int32_t SearchCubic);
void FindMaxCoordinate(int32_t& Row, int32_t& Col, int32_t& Slice, double* pA, int32_t i, int32_t j, int32_t k, int32_t SearchCubic);
Sphere& MinEnergy(Sphere &sph_co, double& alpha, double& beta, double& lamda, double& delta, double& theta, double* V_interp, double* grad_value);
double calc_energy(Sphere &sph_co, int32_t i, int32_t j, double& alpha, double& beta, double& lamda, double& delta, double& theta, double* V_interp, double* grad_value);
inline double calc_energy_elas(Sphere &sph_co, int32_t i, int32_t j, double& alpha);
inline double calc_energy_bend(Sphere &sph_co, int32_t i, int32_t j, double& beta);
inline double AngleBetweenVectors(double a[], double b[]);
double calc_surf(Sphere& sph_co);
inline double area(Point& A, Point& B, Point& C);
inline double polyarea(vector<Point>& poly);
double calc_vol(Sphere& sph_co);
inline double distance_point_to_plane(Point& a, Point& b, Point& c, Point& D);
vector<double> GetValue(Sphere& sph_co, double* V_interp, IFSliceBetweenSlice I);
vector<Point> interp1(vector<Point>& input, int32_t interp_n);
double Func_mean(vector<double>& input);
double Func_std(vector<double>& input, double& mean);
double Func_skewness(vector<double>& input, double& mean, double& std, double& const_s_m_n);
double Func_kurtosis(vector<double>& input, double& mean, double& std, double& const_s_m_n);
double Func_sphericity(double volume, double surface);

 






