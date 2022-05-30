
/*
Ali ÖZTÜRK, a.ozturk@alparslan.edu.tr
30.05.2022
If you use the code, plase cite our paper,


A. Ozturk, I. Cayiroglu, "A Real-Time Implementation of Singular Spectrum Analysis to Object Tracking with SIFT", ETASR. 
*/
#include <iostream>
#include <string>
#include <stdio.h>


#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/StdVector>

#ifdef _WIN32
#include "win_timer.h"
using win32::Stopwatch;
Stopwatch sw;
#else
#include "timer.h"
double iStart, iElaps;
#endif // _WIN32


using namespace cv;
using namespace std;
using Eigen::MatrixXf;
using Eigen::JacobiSVD;

struct ssaFilter
{
	vector<float> y;
	float eigenPercantage;

};

//The original SSA Algorithm
struct ssaFilter SSA(const vector<float>& data,unsigned int L,unsigned int index);
//The modified  Algorithm for Real Time SSA (The reconstruction step is only modified) 
vector<float>  RT_SSA(const vector<vector<float>>& data,unsigned int L,unsigned int index);

// median of calculated execution time vector 
double calculate_median(std::vector<double> v);


int main()
{
	// Load the raw data from memory
	// File : X Y t  
	// X and Y are pixel coordinates, and t recorded time 
	const char* filename = "raw_data.dat";
	

	int bufferLength = 5000;
	char buffer[5000];

	FILE* fp = fopen(filename, "r");

	if (!fp) {
		printf("Cant open file\n");
		return -1;
	}

	float x = 0.0,y=0.0,t=0.0;
	vector<float> X;
	vector<float> Y;
	vector<float> T;

	while (fgets(buffer, bufferLength, fp)) {
		//printf("%s\n", buffer);   
		if (3== sscanf(buffer, "%f %f %f", &x,&y,&t)) {

			X.push_back(x);
			Y.push_back(y);
			T.push_back(t);
		}
	}

	fclose(fp);

    //Create a file to record the filtered data.
	string saveName = "filtered_data.txt";
	if (remove(saveName.c_str()) == 0)
	{
		printf("\n    %s is deleted successfully.\n", saveName.c_str());

	}
	FILE* fp_s = fopen(saveName.c_str(), "a");

	
	// Here after, algorithm of the real time SSA starts. 
    
	//Parameter of SSA 
	unsigned int L=10; //Window Length
	unsigned int index=1; //Reconstruction parameter (r)

	ssaFilter S;
	std::vector<double> v;
	double median_timeEllapse;

	vector<vector<float>> buf(2); // Two dimensional buffer for two coordinates (X and Y)
	vector<float> filter;

	short frame_delay=40; //Buffer size


    iStart = seconds();

	vector<float> filtered_frame(1);
	for(unsigned int i=0;i<X.size();i++)
	{


     //Fill the buffers
     if(buf[0].size()<frame_delay)
	 {
		 buf[0].push_back(X[i]); // x-coordinate
		 buf[1].push_back(Y[i]); // y-coordinate
		 
		
	 }


     //Filter the buffer as new frame arrives
	 if(buf[0].size()==frame_delay)
	 {
		 filtered_frame=RT_SSA(buf,L,index);
		 buf[0].erase(buf[0].begin()); // Delete the last entry from the buffer
		 buf[0].push_back(X[i]); // x-coordinate
		 buf[1].erase(buf[1].begin());// Delete the last entry from the buffer
		 buf[1].push_back(Y[i]); // y-coordinate

		 fprintf(fp_s,"%f %f\n",filtered_frame[0],filtered_frame[1]); //Write filtered data into the file

	 }	

	}
	iElaps = seconds() - iStart;

	cout<<iElaps * 1000<<endl;

	fclose(fp_s);

	return 0;
}

vector<float>  RT_SSA(const vector<vector<float>>& data,unsigned int L,unsigned int index)
{
	short n=data.size(); 
	vector<float> last(n);
    
	unsigned int N=data[0].size();
	unsigned int K=N-L+1;
	unsigned int i,j,k;

	for (k=0;k<n;k++)
	{
		MatrixXf X=MatrixXf::Zero(L, K);

		// Step 1 (Embedding)	
		for(unsigned int i=0;i<K;i++)
		{
			for(unsigned int j=0;j<L;j++)
			{
				unsigned int offset=i+j;
				X(j,i)=data[k][offset];
			}
		}

		//%%%%%%%%% Step-2 (SVD) %%%%%%%%%%%%%%%%%%%
		MatrixXf X_transpose = X.transpose();
		Eigen::MatrixXf SS=X*X_transpose;
		Eigen::JacobiSVD<MatrixXf> svd(SS,Eigen::ComputeThinU| Eigen::ComputeThinV);
		Eigen::VectorXf Lamdas= svd.singularValues();

	    Eigen::MatrixXf U=svd.matrixU();
	    Eigen::MatrixXf V=svd.matrixV();

		MatrixXf VV = MatrixXf::Zero(K, L);

		vector<MatrixXf> XX(L);
		
		for(unsigned int i=0;i<L;i++)
		{
			VV.col(i)=X_transpose*U.col(i)/std::sqrt(Lamdas(i));
			Eigen::MatrixXf VV_transpose=VV.col(i).transpose();
			XX[i]=std::sqrt(Lamdas(i))*U.col(i)*VV_transpose;      
		}

		// Grouping
	    MatrixXf X_C = MatrixXf::Zero(L, K);
		
		for(unsigned int i=0;i<index;i++)
		{
			X_C+=XX[i];
		}

		//Reconstruction
		last[k]=X_C(L-1,K-1);
	}
	


	return last;

}


struct ssaFilter SSA(const vector<float>& data,unsigned int L,unsigned int index)
{
	ssaFilter S;
    // Step 1 (Embedding)
	unsigned int N=data.size();
	unsigned int K=N-L+1;

	MatrixXf X = MatrixXf::Zero(L, K);

	for(unsigned int i=0;i<K;i++)
	{

		for(unsigned int j=0;j<L;j++)
		{
			unsigned int offset=i+j;
			X(j,i)=data[offset];
		}
	}

	//%%%%%%%%% Step-2 (SVD) %%%%%%%%%%%%%%%%%%%
	MatrixXf X_transpose = X.transpose();
	Eigen::MatrixXf SS=X*X_transpose;

	Eigen::JacobiSVD<MatrixXf> svd(SS,Eigen::ComputeThinU| Eigen::ComputeThinV);

	Eigen::VectorXf Lamdas= svd.singularValues();



	S.eigenPercantage=Lamdas(0)/Lamdas.sum();
	Eigen::MatrixXf U=svd.matrixU();
	Eigen::MatrixXf V=svd.matrixV();

	MatrixXf VV = MatrixXf::Zero(K, L);

	vector<MatrixXf> XX(L);

	for(unsigned int i=0;i<L;i++)
	{
		VV.col(i)=X_transpose*U.col(i)/std::sqrt(Lamdas(i));
		Eigen::MatrixXf VV_transpose=VV.col(i).transpose();
		XX[i]=std::sqrt(Lamdas(i))*U.col(i)*VV_transpose;      
	}

	// Grouping
	MatrixXf X_C = MatrixXf::Zero(L, K);
	for(unsigned int i=0;i<index;i++)
	{
		X_C+=XX[i];
	}

   // Reconstruction
	unsigned int LL=std::min(L,K);
	unsigned int KK=std::max(L,K);



	std::vector<float> y(N);
    memset(y.data(), 0, sizeof(float)*N);

	float scl;

	// Top diagonal
    for(unsigned int i=0;i<LL-1;i++)
	{
		scl=1.0/(float)(i+1);
		for(unsigned int j=0;j<i+1;j++)
		{
			
			y[i]+=scl*X_C(j,i-j);
		}
	}

	scl=1.0/(float)LL;

	//Full diagonal
	for (unsigned int i=LL-1;i<KK;i++)
	{
		for(unsigned int j=0;j<LL;j++)
		{
			y[i]+=scl*X_C(j,i-j);
		}
	}

	//Tile Diagonal
	for(unsigned int i=KK;i<N;i++)
	{
		scl=1.0/(float)(N-i);
		for(unsigned int j=i-KK+1;j<=N-KK;j++)
		{
			y[i]+=scl*X_C(j,i-j);
		}
	}

	S.y=y;

	return S;

}

// median of calculated execution time vector 
double calculate_median(std::vector<double> v)
{
	double median;

	sort(v.begin(), v.end());
	if (v.size() % 2 == 0)
		median = (v[v.size() / 2 - 1] + v[v.size() / 2]) / 2;
	else
		median = v[v.size() / 2];
	return median;
}
