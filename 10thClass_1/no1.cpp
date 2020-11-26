#include <iostream>
#include <opencv2/opencv.hpp>
#include<math.h>
#include<vector>
using namespace cv;
using namespace std;

int calcHOG(cv::Mat src, float* hist, int nAngle, int cellSize)
{
	//参数设置
	int nX = src.cols / cellSize;
	int nY = src.rows / cellSize;

	cv::Mat refMat;
	cvtColor(src, refMat, CV_BGR2GRAY);
	Mat gx, gy;
	Mat mag, angle;
	float binAngle = 360 / nAngle;

	Sobel(src, gx, CV_32F, 1, 0, 1);
	Sobel(src, gy, CV_32F, 0, 1, 1);
	//x方向梯度，y方向梯度，梯度角度决定输出弧度or角度
	cartToPolar(gx, gy, mag, angle, true);

	Rect roi;
	roi.x = 0;
	roi.y = 0;
	roi.width = cellSize;
	roi.height = cellSize;
	for (int i = 0; i < nY; i++)
	{
		for (int j = 0; j < nX; j++)
		{
			cv::Mat roiMat;
			cv::Mat roiMag;
			cv::Mat roiAgl;
			roi.x = j * cellSize;
			roi.y = i * cellSize;
			//赋值图像
			roiMat = src(roi);
			roiMag = mag(roi);
			roiAgl = angle(roi);
			//当前cell第一个元素在数组中的位置
			int head = (i * nX + j) * nAngle;
			for (int n = 0; n < roiMat.rows; n++)
			{
				for (int m = 0; m < roiMat.cols; m++)
				{
					int pos = (int)(roiAgl.at<float>(n, m) / binAngle);
					hist[head + pos] += roiMag.at<float>(n, m);
				}
			}
		}
	}
	return 0;
}
float normL2(float*hist1, float*hist2, int bins)
{
	float sum = 0;
	for (int i = 0; i < bins; i++)
	{
		sum += (hist1[i] - hist2[i]) * (hist1[i] - hist2[i]);
	}
	sum = sqrt(sum);
	return sum;
}
int main()
{
	Mat refMat = imread("C:/Users/hp/Desktop/hogTemplate.jpg");
	Mat plMat= imread("C:/Users/hp/Desktop/img1.jpg");
	Mat bgMat= imread("C:/Users/hp/Desktop/img2.jpg");

	int blockSize = 16;
	int nAngle = 8;
	int nX = refMat.cols / blockSize;
	int nY = refMat.rows / blockSize;

	int bins = nX * nY*nAngle;
	float*ref_hist = new float[bins];
	memset(ref_hist, 0, sizeof(float)*bins);
	float*pl_hist = new float[bins];
	memset(pl_hist, 0, sizeof(float)*bins);
	float*bg_hist = new float[bins];
	memset(bg_hist, 0, sizeof(float)*bins);

	int reCode = 0;
	//计算三张输入图片的HOG
	reCode = calcHOG(refMat, ref_hist, nAngle, blockSize);
	reCode = calcHOG(plMat, pl_hist, nAngle, blockSize);
	reCode = calcHOG(bgMat, bg_hist, nAngle, blockSize);

	if(reCode != 0) {
		return -1;
	}
	//计算直方图距离
	float dis1 = normL2(ref_hist, pl_hist, bins);
	float dis2 = normL2(ref_hist, bg_hist, bins);
	std::cout << "distance between reference and img1:" << dis1 << std::endl;
	std::cout << "distance between reference and img2:" << dis2 << std::endl;

	if (dis1 > dis2)
	{
		std::cout << "img2与原图相似度更高" << std::endl;
	}
	else
	{
		std::cout << "img1与原图相似度更高" << std::endl;
	}

	delete[] ref_hist;
	delete[] pl_hist;
	delete[] bg_hist;
	waitKey(0);
	return 0;


}

