#pragma once
#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
#include<memory>
#include<vector>
#include<numeric>
#include"LD.h"
#include<io.h>
#include<set>
#include<map>
#include<stack>
#include<math.h>

#include "dcmtk\dcmdata\dctk.h"
#include "dcmtk\config\osconfig.h"
#include "DCMTK\dcmimgle\dcmimage.h"
#include "opencv2\opencv.hpp"

using namespace std;
//"C:\\Users\\yzw\\Desktop\\data\\LIDC-IDRI-0001\\*.dcm"
short* GenerateVFromFolder(string folder)
{
	/*vector<vector<short>> img;*/
	vector<string> files;
	vector<int> InstanceNum;
	map<int, int> InstanceNum_s;
	
	FindDicomFile(folder + "\\*.dcm", files);
	auto VSliceNum = files.size();
	unsigned short* pImgV = new unsigned short [512 * 512 * VSliceNum];
	for (unsigned int i = 0; i < VSliceNum; i++)
	{
		shared_ptr<DcmFileFormat> dfile(new DcmFileFormat());
		string str = folder + "\\" + files[i];
		dfile->loadFile(str.c_str());
		DcmDataset* data = dfile->getDataset();
		DcmElement* pElement = nullptr;
		Sint32 pInstanceNum = 0;

		if (data->findAndGetElement(DCM_InstanceNumber, pElement) == EC_Normal)
		{			
			pElement->getSint32(pInstanceNum);
			InstanceNum.push_back(pInstanceNum);
		}
	}


	for (unsigned int i = 0; i < VSliceNum; i++)
		InstanceNum_s.insert(pair<int, int>(InstanceNum[i], i));



	for (unsigned int i = 0; i < VSliceNum; i++)
	{
		shared_ptr<DcmFileFormat> dfile(new DcmFileFormat());
		string str = folder + "\\" + files[InstanceNum_s[i+1]];
		
		dfile->loadFile(str.c_str());
		DcmDataset* data = dfile->getDataset();
		DcmElement* pElement = nullptr;
		if (data->findAndGetElement(DCM_PixelData, pElement) == EC_Normal)
		{
			Uint16* temp = nullptr;
			pElement->getUint16Array(temp);
			for (auto j = 0; j < 512 * 512; j++)
				*(pImgV + i * 512 * 512 + j) = *(temp + j);
		}
		
	}

	short* pImgV_c = static_cast<short*>(static_cast<void*>(pImgV));
	auto VInterpSliceNum = VSliceNum * 2 - 1;
	short* pImgInterpV = new short [512 * 512 * VInterpSliceNum];


	for (auto i = 0; i < VInterpSliceNum; i++)
	{
		if (i % 2 == 0)
		{
			for (auto j = 0; j < 512 * 512; j++)
				*(pImgInterpV + i * 512 * 512 + j) = *(pImgV_c + (i / 2) * 512 * 512 + j);
		}
			
		for (auto j = 0; j < 512 * 512; j++)
			*(pImgInterpV + i * 512 * 512 + j) = (*(pImgV_c + ((i - 1) / 2) * 512 * 512 + j) + *(pImgV_c + ((i + 1) / 2) * 512 * 512 + j)) / 2;

	}

	delete [] pImgV;


	return pImgInterpV;
}

int FindDicomFile(string path, vector<string>& files)
{
	intptr_t handle;                                                //用于查找的句柄
	struct _finddata_t fileinfo;							//文件信息的结构体
	handle = _findfirst(path.c_str(), &fileinfo);         //第一次查找
	if (-1 == handle)return -1;
	do
	{
		files.push_back(fileinfo.name);
	}												//打印出找到的文件的文件名
	while (!_findnext(handle, &fileinfo));            //循环查找其他符合的文件，知道找不到其他的为止
	_findclose(handle);
	return 0;										//别忘了关闭句柄
}

void RegionGrow_3D(double * const input, double * const output, int SliceNum, Point SeedPoint, int Threshold)
{
	long long count = 0;
	shared_ptr<double> label(new double[512 * 512 * SliceNum]);
	auto pLabel = label.get();
	for (long long i = 0; i < 512 * 512 * SliceNum; i++)//output初始化为零
		*(output + i) = 0;
	for (long long i = 0; i < 512 * 512 * SliceNum; i++)//Label初始化为零
		*(pLabel + i) = 0;
	stack<Point> SeedStack;
	SeedStack.push(SeedPoint);
	

	while (!SeedStack.empty()) 
	{

		Point P = {0,0,0};
		P = SeedStack.top();
		SeedStack.pop();

		/*cout << P.RowNum << " " << P.ColNum << " " << P.SliceNum << endl;*/
		*(pLabel + 512 * 512 * (P.SliceNum - 1) + 512 * (P.RowNum - 1) + P.ColNum - 1) = 1;
		if (*(input + 512 * 512 * (P.SliceNum - 1) + 512 * (P.RowNum - 1) + P.ColNum - 1) <= Threshold)
		{
			count++;
			*(output + 512 * 512 * (P.SliceNum - 1) + 512 * (P.RowNum - 1) + P.ColNum - 1) = 1;
			if( (P.SliceNum > 1 && P.SliceNum < SliceNum) && (P.RowNum > 1 && P.RowNum < 512) && (P.ColNum > 1 && P.ColNum < 512) )
			{
				struct Point Left = { P.RowNum, P.ColNum - 1 , P.SliceNum };

				struct Point Up = { P.RowNum - 1, P.ColNum , P.SliceNum };

				struct Point Right = { P.RowNum, P.ColNum + 1 , P.SliceNum };

				struct Point Down = { P.RowNum + 1, P.ColNum, P.SliceNum };

				struct Point Above = { P.RowNum, P.ColNum , P.SliceNum + 1 };

				struct Point Below = { P.RowNum, P.ColNum , P.SliceNum - 1 };
				if (*(pLabel + 512 * 512 * (Left.SliceNum - 1) + 512 * (Left.RowNum - 1) + Left.ColNum - 1) == 0)
					SeedStack.push(Left);

				if (*(pLabel + 512 * 512 * (Up.SliceNum - 1) + 512 * (Up.RowNum - 1) + Up.ColNum - 1) == 0)
					SeedStack.push(Up);

				if (*(pLabel + 512 * 512 * (Right.SliceNum - 1) + 512 * (Right.RowNum - 1) + Right.ColNum - 1) == 0)
					SeedStack.push(Right);

				if (*(pLabel + 512 * 512 * (Down.SliceNum - 1) + 512 * (Down.RowNum - 1) + Down.ColNum - 1) == 0)
					SeedStack.push(Down);

				if (*(pLabel + 512 * 512 * (Above.SliceNum - 1) + 512 * (Above.RowNum - 1) + Above.ColNum - 1) == 0)
					SeedStack.push(Above);

				if (*(pLabel + 512 * 512 * (Below.SliceNum - 1) + 512 * (Below.RowNum - 1) + Below.ColNum - 1) == 0)
					SeedStack.push(Below);
			}
		}
	}

}

void Gradient_3D(double * const input, double * const output, int SliceNum)
{
	for (long long i = 0; i < 512 * 512 * SliceNum; i++)//output初始化为零
		*(output + i) = 0;
	for (auto i = 1; i < SliceNum - 1; i++)
		for (auto j = 1; j < 512 - 1; j++)
			for (auto k = 1; k < 512 - 1; k++)
			{
				double Diff_Slice = (*(input + 512 * 512 * (i - 1) + 512 * j + k) - *(input + 512 * 512 * (i + 1) + 512 * j + k)) / 2;
				double Diff_Row = (*(input + 512 * 512 * i + 512 * (j - 1) + k) - *(input + 512 * 512 * i + 512 * (j + 1) + k)) / 2;
				double Diff_Col = (*(input + 512 * 512 * i + 512 * j + k - 1) - *(input + 512 * 512 * i + 512 * j + k + 1)) / 2;
				*(output + 512 * 512 * i + 512 * j + k) = sqrt(Diff_Slice*Diff_Slice + Diff_Row*Diff_Row + Diff_Col*Diff_Col);
				//if (i == 150 && j == 222 && k == 100)
				//{
				//	cout << *(input + 512 * 512 * (i + 1) + 512 * j + k) << " " << *(input + 512 * 512 * (i - 1) + 512 * j + k) << endl;
				//	cout << Diff_Slice << " " << Diff_Row << " " << Diff_Col << endl;
				//}
			}
}

void Morphology(double * const input, double * const output, int32_t SliceNum, int32_t element_size)
{
	cv::Mat element = getStructuringElement(cv::MORPH_CROSS, cv::Size(element_size, element_size), cv::Point(-1, -1));
	for (int32_t i = 0; i < SliceNum; i++)
	{
		cv::Mat src = cv::Mat(RowSize, ColSize, CV_64FC1, input + RowSize*ColSize * i, 0);
		cv::Mat dst = cv::Mat(RowSize, ColSize, CV_64FC1, output + RowSize*ColSize * i, 0);
		cv::morphologyEx(src, dst, cv::MORPH_CLOSE, element);
	}
}

void FindLocalMax(vector<int32_t>& Row, vector<int32_t>& Col, vector<int32_t>& Slice, double * const pA, int SliceNum, int SearchCubic)
{
	for (auto i = SearchCubic; i < 512 - SearchCubic; i++)
		for (auto j = SearchCubic; j < 512 - SearchCubic; j++)
			for (auto k = SearchCubic; k < SliceNum - SearchCubic; k++)
			{
				if ( *(pA + 512 * 512 * k + 512 * i + j) >= 500 )
				{
					int32_t row_i = 0;
					int32_t col_i = 0;
					int32_t slice_i = 0;
					FindMaxCoordinate(row_i, col_i, slice_i, pA, i, j, k, SearchCubic);
					if (row_i == i && col_i == j && slice_i == k)
					{
						Row.push_back(i);
						Col.push_back(j);
						Slice.push_back(k);
					}
				}
			}
}

void FindMaxCoordinate(int32_t & Row, int32_t & Col, int32_t & Slice, double* pA, int32_t i, int32_t j, int32_t k, int32_t SearchCubic)
{
	double nmax = 0;
	Row = i; Col = j; Slice = k;
	for (int32_t l = i - SearchCubic; l < i + SearchCubic; l++)
		for (int32_t m = j - SearchCubic; m < j + SearchCubic; m++)
			for (int32_t n = k - SearchCubic; n < k + SearchCubic; n++)
			{
				if ( *(pA + 512 * 512 * n + 512 * l + m) != 0	&&	*(pA + 512 * 512 * n + 512 * l + m) >= nmax)
				{
					nmax = *(pA + 512 * 512 * n + 512 * l + m); 
					Row = l; 
					Col = m; 
					Slice = n;
				}
			}

}

Sphere& MinEnergy(Sphere& sph_co, double& alpha, double& beta, double& lamda, double& delta, double& theta, double * V_interp, double * grad_value)
{
	int sph_slicenum = sph_co.SliceNum;
	int numperslice = sph_co.NumPerSlice;
	cout << sph_slicenum << " " << numperslice << endl;
	int iter_time = 0;
	int move_length = 1;
	double E_init = 0;
	double E_old = 0;
	double E_new = 0;
	for (auto i = 0; i < sph_slicenum; i++)
		for (auto j = 0; j < numperslice; j++)
		{
			E_init += calc_energy(sph_co, i+1, j+1, alpha, beta, lamda, delta, theta, V_interp, grad_value);
		}
	E_old = E_init;
	E_new = E_old;

	int* t = new int [sph_slicenum*numperslice];
	//shared_ptr<double> smartpoint_t(new double[sph_slicenum*numperslice]);
	//double* t = smartpoint_t.get();
	for (int i = 0; i < sph_slicenum*numperslice; i++)//output初始化为零
		*(t + i) = 0;


	while (E_new <= E_old)
	{
		iter_time++;
		if (iter_time == 100)
			break;

		E_old = E_new;

		for (auto i = 1; i <= sph_slicenum; i++)
			for (auto j = 1; j <= numperslice; j++)
			{
				//Centre
				Sphere sph_co_centre = sph_co;
				Point centre_loc = { sph_co.GetLoc(i, j).RowNum,sph_co.GetLoc(i, j).ColNum,sph_co.GetLoc(i, j).SliceNum };
				sph_co_centre.SetLoc(i, j, centre_loc);
				double E_centre = calc_energy(sph_co_centre, i, j, alpha, beta, lamda, delta, theta, V_interp, grad_value);

				//LeftUp
				Sphere sph_co_leftup = sph_co;
				Point leftup_loc = { sph_co.GetLoc(i, j).RowNum - move_length,sph_co.GetLoc(i, j).ColNum - move_length,sph_co.GetLoc(i, j).SliceNum };
				sph_co_leftup.SetLoc(i, j, leftup_loc);
				double E_leftup = calc_energy(sph_co_leftup, i, j, alpha, beta, lamda, delta, theta, V_interp, grad_value);

				//Up
				Sphere sph_co_up = sph_co;
				Point up_loc = { sph_co.GetLoc(i, j).RowNum - move_length,sph_co.GetLoc(i, j).ColNum,sph_co.GetLoc(i, j).SliceNum };
				sph_co_up.SetLoc(i, j, up_loc);
				double E_up = calc_energy(sph_co_up, i, j, alpha, beta, lamda, delta, theta, V_interp, grad_value);

				//RightUp
				Sphere sph_co_rightup = sph_co;
				Point rightup_loc = { sph_co.GetLoc(i, j).RowNum - move_length,sph_co.GetLoc(i, j).ColNum + move_length,sph_co.GetLoc(i, j).SliceNum };
				sph_co_rightup.SetLoc(i, j, rightup_loc);
				double E_rightup = calc_energy(sph_co_rightup, i, j, alpha, beta, lamda, delta, theta, V_interp, grad_value);

				//Left
				Sphere sph_co_left = sph_co;
				Point left_loc = { sph_co.GetLoc(i, j).RowNum,sph_co.GetLoc(i, j).ColNum - move_length,sph_co.GetLoc(i, j).SliceNum };
				sph_co_left.SetLoc(i, j, left_loc);
				double E_left = calc_energy(sph_co_left, i, j, alpha, beta, lamda, delta, theta, V_interp, grad_value);

				//Right
				Sphere sph_co_right = sph_co;
				Point right_loc = { sph_co.GetLoc(i, j).RowNum,sph_co.GetLoc(i, j).ColNum + move_length,sph_co.GetLoc(i, j).SliceNum };
				sph_co_right.SetLoc(i, j, right_loc);
				double E_right = calc_energy(sph_co_right, i, j, alpha, beta, lamda, delta, theta, V_interp, grad_value);

				//LeftDown
				Sphere sph_co_leftdown = sph_co;
				Point leftdown_loc = { sph_co.GetLoc(i, j).RowNum + move_length,sph_co.GetLoc(i, j).ColNum - move_length,sph_co.GetLoc(i, j).SliceNum };
				sph_co_leftdown.SetLoc(i, j, leftdown_loc);
				double E_leftdown = calc_energy(sph_co_leftdown, i, j, alpha, beta, lamda, delta, theta, V_interp, grad_value);
	
				//Down
				Sphere sph_co_down = sph_co;
				Point down_loc = { sph_co.GetLoc(i, j).RowNum + move_length,sph_co.GetLoc(i, j).ColNum,sph_co.GetLoc(i, j).SliceNum };
				sph_co_down.SetLoc(i, j, down_loc);
				double E_down = calc_energy(sph_co_down, i, j, alpha, beta, lamda, delta, theta, V_interp, grad_value);
	
				//RightDown
				Sphere sph_co_rightdown = sph_co;
				Point rightdown_loc = { sph_co.GetLoc(i, j).RowNum + move_length,sph_co.GetLoc(i, j).ColNum + move_length,sph_co.GetLoc(i, j).SliceNum };
				sph_co_rightdown.SetLoc(i, j, rightdown_loc);
				double E_rightdown = calc_energy(sph_co_rightdown, i, j, alpha, beta, lamda, delta, theta, V_interp, grad_value);

				//Energy Sort
				map<double, int> energy;
				energy.insert(pair<double, int>(E_centre, 1));
				energy.insert(pair<double, int>(E_leftup, 2));
				energy.insert(pair<double, int>(E_up, 3));
				energy.insert(pair<double, int>(E_rightup, 4));
				energy.insert(pair<double, int>(E_left, 5));
				energy.insert(pair<double, int>(E_right, 6));
				energy.insert(pair<double, int>(E_leftdown, 7));
				energy.insert(pair<double, int>(E_down, 8));
				energy.insert(pair<double, int>(E_rightdown, 9));

				auto iterator = energy.begin();
				int min_index = iterator->second;

				switch (min_index)
				{
				case 1:
				{
					if (*(t + (i - 1)*numperslice + j - 1) == 1)
					{
						*(t + (i - 1)*numperslice + j - 1) = 0;
						double u[3] = { 0,0,sph_co.GetLoc(i, 1).SliceNum };
						for (auto m = 0; m < numperslice; m++)
						{
							u[0] += sph_co.GetLoc(i, m + 1).RowNum;
							u[1] += sph_co.GetLoc(i, m + 1).ColNum;
						}
						u[0] /= numperslice;
						u[1] /= numperslice;

						double v1[3] = { u[0] - centre_loc.RowNum ,u[1] - centre_loc.ColNum ,u[2] - centre_loc.SliceNum };
						double v2[3] = { -1,-1,0 };
						double v3[3] = { -1,0,0 };
						double v4[3] = { -1,1,0 };
						double v5[3] = { 0,-1,0 };
						double v6[3] = { 0,1,0 };
						double v7[3] = { 1,-1,0 };
						double v8[3] = { 1,0,0 };
						double v9[3] = { 1,1,0 };
						double ag2 = AngleBetweenVectors(v1, v2);
						double ag3 = AngleBetweenVectors(v1, v3);
						double ag4 = AngleBetweenVectors(v1, v4);
						double ag5 = AngleBetweenVectors(v1, v5);
						double ag6 = AngleBetweenVectors(v1, v6);
						double ag7 = AngleBetweenVectors(v1, v7);
						double ag8 = AngleBetweenVectors(v1, v8);
						double ag9 = AngleBetweenVectors(v1, v9);

						map<double, int> angle;
						angle.insert(pair<double, int>(ag2, 1));
						angle.insert(pair<double, int>(ag3, 2));
						angle.insert(pair<double, int>(ag4, 3));
						angle.insert(pair<double, int>(ag5, 4));
						angle.insert(pair<double, int>(ag6, 5));
						angle.insert(pair<double, int>(ag7, 6));
						angle.insert(pair<double, int>(ag8, 7));
						angle.insert(pair<double, int>(ag9, 8));

						auto iterator2 = angle.begin();
						int min_index2 = iterator->second;

						switch (min_index2)
						{
						case 1:
							sph_co.SetLoc(i, j, leftup_loc); break;
						case 2:
							sph_co.SetLoc(i, j, up_loc); break;
						case 3:
							sph_co.SetLoc(i, j, rightup_loc); break;
						case 4:
							sph_co.SetLoc(i, j, left_loc); break;
						case 5:
							sph_co.SetLoc(i, j, right_loc); break;
						case 6:
							sph_co.SetLoc(i, j, leftdown_loc); break;
						case 7:
							sph_co.SetLoc(i, j, down_loc); break;
						case 8:
							sph_co.SetLoc(i, j, rightdown_loc); break;
						default: cout << "Inside switch Out of Range!" << endl; break;
						}

					}
					else
					{
						*(t + (i - 1)*numperslice + j - 1) = 1;
						sph_co.SetLoc(i, j, centre_loc);
					}
					break;

				}
				case 2:
				{
					*(t + (i - 1)*numperslice + j - 1) = 0;
					sph_co.SetLoc(i, j, leftup_loc); break;
				}
				case 3:
				{
					*(t + (i - 1)*numperslice + j - 1) = 0;
					sph_co.SetLoc(i, j, up_loc); break;
				}
				case 4:
				{
					*(t + (i - 1)*numperslice + j - 1) = 0;
					sph_co.SetLoc(i, j, rightup_loc); break;
				}
				case 5:
				{
					*(t + (i - 1)*numperslice + j - 1) = 0;
					sph_co.SetLoc(i, j, left_loc); break;
				}
				case 6:
				{
					*(t + (i - 1)*numperslice + j - 1) = 0;
					sph_co.SetLoc(i, j, right_loc); break;
				}
				case 7:
				{
					*(t + (i - 1)*numperslice + j - 1) = 0;
					sph_co.SetLoc(i, j, leftdown_loc); break;
				}
				case 8:
				{
					*(t + (i - 1)*numperslice + j - 1) = 0;
					sph_co.SetLoc(i, j, down_loc); break;
				}
				case 9:
				{
					*(t + (i - 1)*numperslice + j - 1) = 0;
					sph_co.SetLoc(i, j, rightdown_loc); break;
				}
				default: cout << "First switch Out of Range!" << endl; break;
				}
			}

		E_new = 0;
		for (auto i = 0; i < sph_slicenum; i++)
			for (auto j = 0; j < numperslice; j++)
			{
				E_new += calc_energy(sph_co, i + 1, j + 1, alpha, beta, lamda, delta, theta, V_interp, grad_value);
			}

	}
	delete [] t;
	cout << "Energy: " << E_new << endl;
	
	return sph_co;
}

double calc_energy(Sphere & sph_co, int i, int j, double& alpha, double& beta, double& lamda, double& delta, double& theta, double * V_interp, double * grad_value)
{
	int sph_slicenum = sph_co.SliceNum;
	int numperslice = sph_co.NumPerSlice;
	double E_elas = 0;
	double E_bend = 0;
	double E_attr = 0;
	double E_grad = 0;
	double E_pot = 0;
	double E = 0;

	E_elas = calc_energy_elas(sph_co, i, j, alpha);
	E_bend = calc_energy_bend(sph_co, i, j, beta);

	//Calculate Geo Centre **********************************************************
	double GeoCentre_Row = 0;
	double GeoCentre_Col = 0;
	double GeoCentre_Slice = sph_co.GetLoc(i, 1).SliceNum;
	for (auto m = 0; m < numperslice; m++)
	{
		GeoCentre_Row += sph_co.GetLoc(i, m + 1).RowNum;
		GeoCentre_Col += sph_co.GetLoc(i, m + 1).ColNum;
	}
	GeoCentre_Row /= numperslice;
	GeoCentre_Col /= numperslice;

	//Calculate Distance ***************************************************************
	shared_ptr<double> smartpoint_pDistance(new double [numperslice]);
	double* pDistance = smartpoint_pDistance.get();

	for (auto m = 0; m < numperslice; m++)
	{
		*(pDistance + m) = sqrt((sph_co.GetLoc(i, m + 1).RowNum - GeoCentre_Row)*(sph_co.GetLoc(i, m + 1).RowNum - GeoCentre_Row) + (sph_co.GetLoc(i, m + 1).ColNum - GeoCentre_Col)*(sph_co.GetLoc(i, m + 1).ColNum - GeoCentre_Col));
	}
	//Calculate DistanceMean ***************************************************************
	double Dmean = 0;
	for (auto m = 0; m < numperslice; m++)
	{
		Dmean += *(pDistance + m);
	}
	Dmean /= numperslice;
	//Calculate DistanceStd ***************************************************************
	double Dstd = 0;
	for (auto m = 0; m < numperslice; m++)
	{
		Dstd += (*(pDistance + m) - Dmean)*(*(pDistance + m) - Dmean);
	}
	Dstd = sqrt(Dstd / (numperslice - 1));
	//Calculate E_attr ***************************************************************
	if (*(pDistance + j - 1) >(Dmean + Dstd))
	{
		E_attr = lamda*(*(pDistance + j - 1)) / Dmean;
	}
	else
	{
		E_attr = 0;
	}
	//Calculate E_grad ***************************************************************
	int ri = sph_co.GetLoc(i, j).RowNum;
	int ci = sph_co.GetLoc(i, j).ColNum;
	int si = sph_co.GetLoc(i, j).SliceNum;
	E_grad = -1 * delta*(*(grad_value + 512 * 512 * (si - 1) + 512 * (ri - 1) + ci - 1));
	//Calculate E_pot ***************************************************************
	E_pot = theta*(*(V_interp + 512 * 512 * (si - 1) + 512 * (ri - 1) + ci - 1));



	E = E_elas + E_bend + E_attr + E_grad + E_pot;
	return E;
}

inline double calc_energy_elas(Sphere & sph_co, int i, int j, double& alpha)
{
	int sph_slicenum = sph_co.SliceNum;
	int numperslice = sph_co.NumPerSlice;
	double E_elas = 0;
	if (i == 1)
	{
		if (j == 1)
		{
			E_elas = 0.5*alpha*(f_1(i, j, i, j + 1) + f_1(i, j, i, numperslice) + f_1(i, j, i, j + 1));
		}
		else if (j == numperslice)
		{
			E_elas = 0.5*alpha*(f_1(i, j, i, j - 1) + f_1(i, j, i, 1) + f_1(i, j, i + 1, j));
		}
		else
		{
			E_elas = 0.5*alpha*(f_1(i, j, i, j - 1) + f_1(i, j, i, j + 1) + f_1(i, j, i + 1, j));
		}
	}
	else if (i == sph_slicenum)
	{
		if (j == 1)
		{
			E_elas = 0.5*alpha*(f_1(i, j, i, j + 1) + f_1(i, j, i, numperslice) + f_1(i, j, i - 1, j));
		}
		else if (j == numperslice)
		{
			E_elas = 0.5*alpha*(f_1(i, j, i, j - 1) + f_1(i, j, i, 1) + f_1(i, j, i - 1, j));
		}
		else
		{
			E_elas = 0.5*alpha*(f_1(i, j, i, j + 1) + f_1(i, j, i, j - 1) + f_1(i, j, i - 1, j));
		}
	}
	else
	{
		if (j == 1)
		{
			E_elas = 0.5*alpha*(f_1(i, j, i, j + 1) + f_1(i, j, i, numperslice) + f_1(i, j, i + 1, j) + f_1(i, j, i - 1, j));
		}
		else if (j == numperslice)
		{
			E_elas = 0.5*alpha*(f_1(i, j, i, j - 1) + f_1(i, j, i, 1) + f_1(i, j, i + 1, j) + f_1(i, j, i - 1, j));
		}
		else
		{
			E_elas = 0.5*alpha*(f_1(i, j, i, j + 1) + f_1(i, j, i, j - 1) + f_1(i, j, i + 1, j) + f_1(i, j, i - 1, j));
		}
	}
	return E_elas;
}

inline double calc_energy_bend(Sphere & sph_co, int i, int j, double& beta)
{
	int sph_slicenum = sph_co.SliceNum;
	int numperslice = sph_co.NumPerSlice;
	double E_bend = 0;
	if (j == 1)
	{
		E_bend = beta*f_2(i, numperslice, i, j, i, j + 1);
	}
	else if (j == numperslice)
	{
		E_bend = beta*f_2(i, j - 1, i, j, i, 1);
	}
	else
	{
		E_bend = beta*f_2(i, j - 1, i, j, i, j + 1);
	}
	return E_bend;
}

inline double AngleBetweenVectors(double a[], double b[])
{
	double Numerator = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
	double Mold_a = sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
	double Mold_b = sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2]);
	double Denominator = Mold_a * Mold_b;
	double t = Numerator / Denominator;
	return acos(t);
}

double calc_surf(Sphere & sph_co)
{
	int sph_slicenum = sph_co.SliceNum;
	int numperslice = sph_co.NumPerSlice;
	double surf_area = 0;
	for (int i = 1; i < sph_slicenum; i++)
		for (int j = 1; j <= numperslice; j++)
		{
			if (j == numperslice)
			{
				Point x1 = sph_co.GetLoc(i, j);
				Point x2 = sph_co.GetLoc(i+1, j);
				Point x3 = sph_co.GetLoc(i, 1);
				Point x4 = sph_co.GetLoc(i+1, j);
				Point x5 = sph_co.GetLoc(i+1, 1);
				Point x6 = sph_co.GetLoc(i, 1);
				surf_area = surf_area + area(x1, x2, x3) + area(x4, x5, x6);
			}
			else
			{
				Point x1 = sph_co.GetLoc(i, j);
				Point x2 = sph_co.GetLoc(i + 1, j);
				Point x3 = sph_co.GetLoc(i, j+1);
				Point x4 = sph_co.GetLoc(i + 1, j);
				Point x5 = sph_co.GetLoc(i + 1, j+1);
				Point x6 = sph_co.GetLoc(i, j+1);
				surf_area = surf_area + area(x1, x2, x3) + area(x4, x5, x6);
			}
		}

	surf_area = surf_area + polyarea(sph_co._d[0]) + polyarea(sph_co._d[sph_slicenum - 1]);
	return surf_area;
}

inline double area(Point & A, Point & B, Point & C)
{
	double X[3] = { B.RowNum - A.RowNum,B.ColNum - A.ColNum,B.SliceNum - A.SliceNum };
	double Y[3] = { C.RowNum - B.RowNum,C.ColNum - B.ColNum,C.SliceNum - B.SliceNum };
	double Z[3] = { X[1] * Y[2] - X[2] * Y[1],X[2] * Y[0] - X[0] * Y[2],X[0] * Y[1] - X[1] * Y[0] };
	return 0.5*sqrt(Z[0]*Z[0]+Z[1]*Z[1]+Z[2]*Z[2]);
}

inline double polyarea(vector<Point>& poly)
{
	double polyarea = 0;
	size_t PointNum = poly.size();
	for (int i = 0; i < PointNum - 2; i++)
	{
		polyarea = polyarea + area(poly[i], poly[i + 1], poly[i + 2]);
	}
	return polyarea;
}

double calc_vol(Sphere & sph_co)
{
	double v = 0;
	int numperslice = sph_co.NumPerSlice;
	int sph_slicenum = sph_co.SliceNum;
	Point cp = sph_co.CentrePoint;
	for (int i = 1; i < sph_slicenum; i++)
		for (int j = 1; j <= numperslice; j++)
		{
			if (j == numperslice)
			{
				Point x1 = sph_co.GetLoc(i, j);
				Point x2 = sph_co.GetLoc(i + 1, j);
				Point x3 = sph_co.GetLoc(i, 1);
				Point x4 = sph_co.GetLoc(i + 1, j);
				Point x5 = sph_co.GetLoc(i + 1, 1);
				Point x6 = sph_co.GetLoc(i, 1);
				double s1 = area(x1, x2, x3);
				double s2 = area(x4, x5, x6);
				double d1 = distance_point_to_plane(x1, x2, x3, cp);
				double d2 = distance_point_to_plane(x4, x5, x6, cp);
				v = v + s1*d1 / 2 + s2*d2 / 2;
			}
			else
			{
				Point x1 = sph_co.GetLoc(i, j);
				Point x2 = sph_co.GetLoc(i + 1, j);
				Point x3 = sph_co.GetLoc(i, j + 1);
				Point x4 = sph_co.GetLoc(i + 1, j);
				Point x5 = sph_co.GetLoc(i + 1, j + 1);
				Point x6 = sph_co.GetLoc(i, j + 1);
				double s1 = area(x1, x2, x3);
				double s2 = area(x4, x5, x6);
				double d1 = distance_point_to_plane(x1, x2, x3, cp);
				double d2 = distance_point_to_plane(x4, x5, x6, cp);
				v = v + s1*d1 / 2 + s2*d2 / 2;
			}
		}

	double s1 = polyarea(sph_co._d[0]);
	double s2 = polyarea(sph_co._d[sph_slicenum -1]);
	Point p1 = sph_co.GetLoc(1, 1);
	Point p2 = sph_co.GetLoc(1, 3);
	Point p3 = sph_co.GetLoc(1, 5);
	Point p4 = sph_co.GetLoc(sph_slicenum, 1);
	Point p5 = sph_co.GetLoc(sph_slicenum, 3);
	Point p6 = sph_co.GetLoc(sph_slicenum, 5);

	double d1 = distance_point_to_plane(p1, p2, p3, cp);
	double d2 = distance_point_to_plane(p4, p5, p6, cp);
	v = v + (s1*d1 + s2*d2) / 2;
	return v;
}

inline double distance_point_to_plane(Point & a, Point & b, Point & c, Point & D)
{
	double d = 0;
	double A[3] = { b.RowNum - a.RowNum,b.ColNum - a.ColNum,b.SliceNum - a.SliceNum };
	double B[3] = { c.RowNum - a.RowNum,c.ColNum - a.ColNum,c.SliceNum - a.SliceNum };
	double C[3] = { D.RowNum - a.RowNum,D.ColNum - a.ColNum,D.SliceNum - a.SliceNum };
	double Z[3] = { A[1] * B[2] - A[2] * B[1],A[2] * B[0] - A[0] * B[2],A[0] * B[1] - A[1] * B[0] };//Z = cross(A,B)
	double norm_Z = sqrt(Z[0] * Z[0] + Z[1] * Z[1] + Z[2] * Z[2]);

	if (norm_Z == 0)
		return 0;
	else
	{
		double t_up = abs(C[0] * Z[0] + C[1] * Z[1] + C[2] * Z[2]);
		return t_up/norm_Z;
	}
}

vector<double> GetValue(Sphere & sph_co, double * V_interp, enum IFSliceBetweenSlice I)
{
	int32_t numperslice = sph_co.NumPerSlice;
	int32_t sph_slicenum = sph_co.SliceNum;
	vector<double> value;
	if (I == NO)//IF SLICE Between SLICE
	{
		for (int32_t i = 0; i < sph_slicenum; i++)
		{
			vector<Point> temp = interp1(sph_co._d[i], 100);
			int32_t PointNum = temp.size();
			for (int32_t j = 0; j < PointNum; j++)
				value.push_back(value(V_interp, temp[i].RowNum, temp[i].ColNum, temp[i].SliceNum));
		}
	}
	return value;
}

vector<Point> interp1(vector<Point>& input, int32_t interp_n)
{
	size_t input_size = input.size();
	size_t section_num = interp_n + 1;
	size_t output_size = input_size * section_num;
	vector<Point> output(output_size);

	// First input_size - 1 Group
	for (size_t i = 0; i < input_size - 1; i++)
	{
		int32_t base_row = input[i].RowNum;
		int32_t base_col = input[i].ColNum;
		int32_t interval_row = (input[i + 1].RowNum - input[i].RowNum)/section_num;
		int32_t interval_col = (input[i + 1].ColNum - input[i].ColNum)/ section_num;
		int32_t Slice = input[i].SliceNum;
		for (size_t j = 0; j < interp_n + 1; j++)
		{
			output[i * section_num + j].RowNum = round(base_row + j*section_num);
			output[i * section_num + j].ColNum = round(base_col + j*section_num);
			output[i * section_num + j].SliceNum = Slice;
		}
	}
	// The Last Group
	int32_t i = input_size - 1;
	int32_t base_row = input[i].RowNum;
	int32_t base_col = input[i].ColNum;
	int32_t interval_row = (input[0].RowNum - input[i].RowNum) / section_num;
	int32_t interval_col = (input[0].ColNum - input[i].ColNum) / section_num;
	int32_t Slice = input[i].SliceNum;
	for (size_t j = 0; j < interp_n + 1; j++)
	{
		output[i * section_num + j].RowNum = round(base_row + j*section_num);
		output[i * section_num + j].ColNum = round(base_col + j*section_num);
		output[i * section_num + j].SliceNum = Slice;
	}
	
	return output;
}

double Func_mean(vector<double>& input)
{
	double sum = accumulate(input.begin(), input.end(), 0.0);
	double mean = sum / input.size();
	return mean;
}

double Func_std(vector<double>& input, double & mean)
{
	double accum = 0.0;
	for_each(input.begin(), input.end(), [&](const double d) {
		accum += (d - mean)*(d - mean);
	});

	double stdn = sqrt(accum / (input.size() - 1));
	return stdn;
}

double Func_skewness(vector<double>& input, double & mean, double & stdn, double & const_s_m_n)
{
	double skewness = 0.0;
	double accum = 0.0;
	std::for_each(input.begin(), input.end(), [&](const double d) {
		skewness += (d - mean)*(d - mean)*(d - mean);
	});

	return skewness/(const_s_m_n*stdn*stdn*stdn);
}

double Func_kurtosis(vector<double>& input, double & mean, double & stdn, double & const_s_m_n)
{
	double kurtosis = 0.0;
	std::for_each(input.begin(), input.end(), [&](const double d) {
		kurtosis += (d - mean)*(d - mean)*(d - mean)*(d - mean);
	});

	return kurtosis / (const_s_m_n*stdn*stdn*stdn*stdn);
}

double Func_sphericity(double volume, double surface)
{
	return pow(PI, static_cast<double>(1) / static_cast<double>(3))*pow(6 * volume, static_cast<double>(2) / static_cast<double>(3)) / surface;
}

void Pol2Cart(double & X, double & Y, double Angle, double Radius)
{
	X = Radius*cos(Angle);
	Y = Radius*sin(Angle);
}




