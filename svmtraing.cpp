#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <vector>
#include <iomanip>

using namespace std;
using namespace cv;

int main(){
	////////////////////////////calculate the input image grey histogram/////////////////////////////////
	char temp[100];
	string filename;
	/// Establish the number of bins
	int histSize = 128;
	/// Set the ranges
	float range[] = { 0, 256 };
	const float* histRange = { range };
	bool uniform = true; bool accumulate = false;
	Mat grey_hist;

	Mat SVM_train_data(0, histSize, CV_32FC1);
	Mat SVM_train_label;
	Mat _SVM_train_data;
	Mat _SVM_train_label;
	//read label 1 images and calculate the image histogram
	for (int i = 1; i <= 99; i++){
		sprintf(temp, "Training\\Train1\\label1\\Train1_%d.jpg", i);
		filename = temp;
		Mat src = imread(filename, CV_BGR2GRAY);

		Mat img;
		src.copyTo(img);
		blur(img, img, Size(3, 3));
		equalizeHist(img, img);
		img = img.reshape(1, 1);
		img.convertTo(img,CV_32F);
		_SVM_train_data.push_back(img);
		_SVM_train_label.push_back(1);

		//equalizeHist(src,src);
		/// Compute the histograms:
		calcHist(&src, 1, 0, Mat(), grey_hist, 1, &histSize, &histRange, uniform, accumulate);
		grey_hist = grey_hist.reshape(1, 1);
		grey_hist.convertTo(grey_hist, CV_32FC1);
		SVM_train_data.push_back(grey_hist);
		SVM_train_label.push_back(1);
	}
	//read lebel 0 images and calculaate the image histogram
	for (int i = 1; i <= 110; i++){
		sprintf(temp, "Training\\Train1\\label0\\Train0_%d.jpg", i); //\Training\Train1
		filename = temp;
		Mat src = imread(filename, CV_BGR2GRAY);

		Mat img;
		src.copyTo(img);
		blur(img, img, Size(3, 3));
		equalizeHist(img, img);
		img = img.reshape(1, 1);
		img.convertTo(img, CV_32F);
		_SVM_train_data.push_back(img);
		_SVM_train_label.push_back(0);

		//equalizeHist(src, src);
		/// Compute the histograms:
		calcHist(&src, 1, 0, Mat(), grey_hist, 1, &histSize, &histRange, uniform, accumulate);
		grey_hist = grey_hist.reshape(1, 1);
		grey_hist.convertTo(grey_hist, CV_32FC1);
		SVM_train_data.push_back(grey_hist);
		SVM_train_label.push_back(0);
	}

	SVM_train_data.convertTo(SVM_train_data, CV_32FC1);
	FileStorage fs("SVM.xml", FileStorage::WRITE);
	fs << "TrainingData" << SVM_train_data;
	fs << "labels" << SVM_train_label;
	fs.release();

	///////////////////////////draw histogram////////////////////////////////////////////////
	//// Draw the histograms for B, G and R
	//int hist_w = 512; int hist_h = 400;
	//int bin_w = cvRound((double)hist_w / histSize);
	//Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(0));
	///// Normalize the result to [ 0, histImage.rows ]
	//normalize(grey_hist, grey_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	///// Draw for each channel
	//for (int i = 1; i < histSize; i++)
	//{
	//	line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(grey_hist.at<float>(i - 1))),
	//		Point(bin_w*(i), hist_h - cvRound(grey_hist.at<float>(i))),
	//		Scalar(255), 2, 8, 0);
	//}
	///// Display
	//namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE);
	//imshow("calcHist Demo", histImage);
	////////////////////////////////////////////end///////////////////////////////////////

	//Setting up SVM parameters
	CvSVMParams SVM_params;
	SVM_params.svm_type = CvSVM::C_SVC;
	SVM_params.kernel_type = CvSVM::LINEAR; //CvSVM::LINEAR;  
	SVM_params.degree = 0;
	SVM_params.gamma = 1;
	SVM_params.coef0 = 0;
	SVM_params.C = 1;
	SVM_params.nu = 0;
	SVM_params.p = 0;
	SVM_params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 1000, 0.01);
	//Train SVM  
	CvSVM svmClassifier(SVM_train_data, SVM_train_label, Mat(), Mat(), SVM_params);

	//Setting up SVM parameters/////////////////////image as input/////////////////////
	CvSVMParams _SVM_params;
	_SVM_params.svm_type = CvSVM::C_SVC;
	_SVM_params.kernel_type = CvSVM::LINEAR; //CvSVM::LINEAR;  
	_SVM_params.degree = 0;
	_SVM_params.gamma = 1;
	_SVM_params.coef0 = 0;
	_SVM_params.C = 1;
	_SVM_params.nu = 0;
	_SVM_params.p = 0;
	_SVM_params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 1000, 0.01);
	//Train SVM  
	CvSVM _svmClassifier(_SVM_train_data, _SVM_train_label, Mat(), Mat(), SVM_params);

	//test the error rate for plate detectcion
	Mat SVM_test_label;
	Mat SVM_test_data(0, histSize, CV_32FC1);
	Mat groundTruth(0, 1, CV_32FC1);
	int count1 = 0, count2 = 0;
	int a = 0, b = 0;
	for (int i = 1; i <= 29; i++){
		sprintf(temp, "Testing\\Test1\\label1\\Test1_%d.jpg", i);
		filename = temp;
		Mat test = imread(filename, CV_BGR2GRAY);

		Mat img;
		test.copyTo(img);
		blur(img, img, Size(3, 3));
		equalizeHist(img, img);
		img = img.reshape(1, 1);
		img.convertTo(img, CV_32F);
		int _response = svmClassifier.predict(grey_hist);
		cout << "1 " << _response << endl;
		if (_response!=1)
			a++;

		//equalizeHist(test, test);
		calcHist(&test, 1, 0, Mat(), grey_hist, 1, &histSize, &histRange, uniform, accumulate);
		grey_hist = grey_hist.reshape(1, 1);
		grey_hist.convertTo(grey_hist, CV_32FC1);
		SVM_test_data.push_back(grey_hist);
		int response = svmClassifier.predict(grey_hist);
		SVM_test_label.push_back(response);
		groundTruth.push_back(1);
		if (response != 1)
			count1++;
		cout << "correct label is 1, svm predict label is: " << response << endl;
	}

	for (int i = 1; i <= 87; i++){
		sprintf(temp, "Testing\\Test1\\label0\\Test0_%d.jpg", i);
		filename = temp;
		Mat test = imread(filename, CV_BGR2GRAY);

		Mat img;
		test.copyTo(img);
		blur(img, img, Size(3, 3));
		equalizeHist(img, img);
		img = img.reshape(1, 1);
		img.convertTo(img, CV_32F);
		int _response = svmClassifier.predict(grey_hist);
		cout << "0 " << _response << endl;
		if (_response != 0)
			b++;

		//equalizeHist(test, test);
		calcHist(&test, 1, 0, Mat(), grey_hist, 1, &histSize, &histRange, uniform, accumulate);
		grey_hist = grey_hist.reshape(1, 1);
		grey_hist.convertTo(grey_hist, CV_32FC1);
		int response = svmClassifier.predict(grey_hist);
		SVM_test_data.push_back(grey_hist);
		SVM_test_label.push_back(response);
		groundTruth.push_back(0);
		if (response != 0)
			count2++;
		cout << "correct label is 0, svm predict label is: " << response << endl;
	}


	double errorRate;
	//calculate the number of unmatched classes  
	errorRate = (double)countNonZero(groundTruth - SVM_test_label) / SVM_test_data.rows;
	cout << "Total error rate is: " << errorRate << endl;
	cout << "label 1 -> 0 count is " << count1 << "/29 error rate is " << double(count1) / 29 << endl;
	cout << "label 0 -> 1 count is " << count2 << "/87 error rate is " << double(count2) / 87 << endl;
	cout << a << double(a) / 29 << endl;
	cout << b << double(b) / 87 << endl;
	waitKey(0);
}