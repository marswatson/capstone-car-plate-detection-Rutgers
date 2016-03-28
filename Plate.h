#ifndef PLATE_H
#define PLATE_H

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;


class Plate{
public:
	Plate();
	Plate(Mat img);
	void PlateDetection();
	bool verifySizes(RotatedRect mr);
	vector< Mat > PlateResults;
private:
	vector<Rect> position;
	Mat img_gray,img_org;
};

#endif