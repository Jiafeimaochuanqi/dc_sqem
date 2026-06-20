#pragma once
#include "cnpy.h"
#include"NumpyUtil.h"
#include<iostream>
#include<string>



#include <Eigen/Dense>


using namespace std;
using namespace Eigen;
using namespace cnpy;

typedef Eigen::Matrix<float, Dynamic, Dynamic, RowMajor> MatF;
typedef Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> MatF3;
typedef Eigen::Matrix<double, Dynamic, Dynamic, RowMajor> MatD;
typedef Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> MatD3;
typedef Eigen::Matrix<int, Dynamic, Dynamic, RowMajor> MatI;
typedef Eigen::Matrix<float, Dynamic, Dynamic, ColMajor> MatFC;
typedef Eigen::Matrix<int, Dynamic, Dynamic, ColMajor> MatIC;
typedef Eigen::Matrix<bool, Dynamic, Dynamic, RowMajor> MatB;
typedef Eigen::Array<bool, Dynamic, 1> ArrayB1;


inline Map<ArrayB1> ToArrayB1(NpyArray &np)
{
	if (np.data_holder == nullptr)
	{
		throw "No Data";
	}

	auto shape = np.shape;
	auto L = np.shape.size();
	if (L > 1)
	{
		cout << "Error" << endl;
	}

	return Map<ArrayB1>(np.data<bool>(), shape[0], 1);
}


inline Map<MatB> ToEigenB(NpyArray &np)
{
	if (np.data_holder == nullptr)
	{
		throw "No Data";
	}

	auto shape = np.shape;
	auto L = np.shape.size();
	if (L > 2)
	{
		cout << "Error" << endl;
	}

	if (L == 1)
	{
		return Map<MatB>(np.data<bool>(), shape[0], 1);
	}
	else
	{
		return Map<MatB>(np.data<bool>(), shape[0], shape[1]);
	}
}

inline Map<MatD> ToEigenD(NpyArray &np)
{
	if (np.data_holder == nullptr)
	{
		throw "No Data";
	}

	auto shape = np.shape;
	auto L = np.shape.size();
	if (L > 2)
	{
		cout << "Error" << endl;
	}

	if (L == 1)
	{
		return Map<MatD>(np.data<double>(), shape[0], 1);
	}
	else
	{
		return Map<MatD>(np.data<double>(), shape[0], shape[1]);
	}
}

inline Map<MatF> ToEigen(NpyArray &np)
{
	if (np.data_holder == nullptr)
	{
		throw "No Data";
	}

	auto shape = np.shape;
	auto L = np.shape.size();
	if (L > 2)
	{
		cout << "Error" << endl;
	}

	if (L == 1)
	{
		return Map<MatF>(np.data<float>(), shape[0], 1);
	}
	else
	{
		return Map<MatF>(np.data<float>(), shape[0], shape[1]);
	}
}

inline MatF ToEigenC(NpyArray &np)
{
	if (np.data_holder == nullptr)
	{
		throw "No Data";
	}

	auto shape = np.shape;
	auto L = np.shape.size();
	if (L > 2)
	{
		cout << "Error" << endl;
	}

	if (L == 1)
	{
		return Map<MatFC>(np.data<float>(), shape[0], 1);
	}
	else
	{
		return Map<MatFC>(np.data<float>(), shape[0], shape[1]);
	}
}

inline Map<MatI> ToEigenInt(NpyArray &np)
{
	if (np.data_holder == nullptr)
	{
		throw "No Data";
	}

	auto shape = np.shape;
	auto L = np.shape.size();
	if (L > 2)
	{
		cout << "Error" << endl;
	}

	if (L == 1)
	{
		return Map<MatI>(np.data<int>(), shape[0], 1);
	}
	else
	{
		return Map<MatI>(np.data<int>(), shape[0], shape[1]);
	}
}

inline MatF Slice(MatF A, vector<int> indexes)
{

	MatF Y = MatF::Zero(indexes.size() * 3, A.cols());
	for (size_t i = 0; i < indexes.size(); i++)
	{
		auto idx = indexes[i];
		Y.row(3 * i + 0) = A.row(3 * idx + 0);
		Y.row(3 * i + 1) = A.row(3 * idx + 1);
		Y.row(3 * i + 2) = A.row(3 * idx + 2);
	}

	return Y;
}

inline vector<int> ToVector(MatI &mat)
{
	vector<int> vec(mat.data(), mat.data() + mat.size());
	return vec;
}

inline Map<MatF> Reshape(MatF &vec, int cols)
{
	return Map<MatF>((float*)vec.data(), Eigen::Index(vec.size() / cols), Eigen::Index(cols));
}
