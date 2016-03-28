#include "Plate.h"
#include "OCR.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <iostream>  
#include <fstream> 
#include <string> 
#include <math.h>


using namespace std;
using namespace cv;

int main(){
	////////////////////////////////////extract training data//////////////////////////////////////////////
	string filename;
	vector<Plate> plate;
	//input the Image
	for (int i = 1; i <= 1; i++){
		cout << i << endl;
		char temp[20];
		sprintf(temp,"Cars\\Img%d.jpg",i);
		filename = temp;
		Mat img = imread(filename);
		Plate p(img);
		plate.push_back(p);
	}

	//store all the rectangle
	vector<Mat> Traing_Data;
	for (int i = 0; i < plate.size(); i++){
		for (int j = 0; j < plate[i].PlateResults.size(); j++)
			Traing_Data.push_back(plate[i].PlateResults[j]);
	}

	for (int i = 0; i < Traing_Data.size(); i++){
		//cout << i << endl;
		stringstream ss(stringstream::in | stringstream::out);
		ss << "Result\\" <<"result" << "_" << i << ".jpg";
		imwrite(ss.str(), Traing_Data[i]);
	}
	///////////////////////////////////////end///////////////////////////////////////////

	////////////////////////////////OCR Test/////////////////////////////////////////////
	//vector<OCR> ocr_plate;
	//for (int i = 1; i < 129; i++){
	//	char temp[50];
	//	sprintf(temp, "OCR-TestImage\\plate%d.jpg", i);
	//	string filename = temp;
	//	Mat img = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	//	OCR ocr_temp(img);
	//	ocr_temp.Segment();
	//	ocr_plate.push_back(ocr_temp);
	//}
	//
	////store the charaters
	//for (int i = 0; i < ocr_plate.size(); i++){
	//	char temp[30];
	//	for (int j = 0; j < ocr_plate[i].Chars.size(); j++){
	//		sprintf(temp, "Result//plate%d-%d.jpg", i,j);
	//		imwrite(temp, ocr_plate[i].Chars[j]);
	//	}
	//}

	return 0;
}