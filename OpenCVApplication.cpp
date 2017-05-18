// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <vector>

typedef struct{
	int step;
	int constantM;
	vector<vector<int>> labels;
	vector<vector<double>> distances;
	vector<int> occurrencesCluster;
	vector<vector<double>> clusters;
}Slic;

int k_slider;
int m_slider;
char fname[MAX_PATH];

void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}


void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}


void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		imshow("source", frame);
		imshow("gray", grayFrame);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i<hist_cols; i++)
	if (hist[i] > max_hist)
		max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

Point findMinumum(Mat img, Point center){
	double min = FLT_MAX;
	Point min_point = Point(center.x, center.y);

	for (int i = -1; i <= 1; i++){
		for (int j = -1; j <= 1; j++){
			Vec3d p1 = img.at<Vec3d>(center.x + i + 1, center.y + j);
			Vec3d p2 = img.at<Vec3d>(center.x + i, center.y + j + 1);
			Vec3d p3 = img.at<Vec3d>(center.x + i, center.y + j);

			double c1 = (p1[0] + p1[1] + p1[2]) / 3.0;
			double c2 = (p2[0] + p2[1] + p2[2]) / 3.0;
			double c3 = (p3[0] + p3[1] + p3[2]) / 3.0;

			if (sqrt(pow(c1 - c3, 2.0)) + sqrt(pow(c2 - c3, 2.0)) < min){
				min = fabs(c1 - c3) + fabs(c2 - c3);
				min_point.x = center.x + i;
				min_point.y = center.y + j;
			}

		}
	}

	return min_point;
}

void initialize(Mat src, int superpixelNr, Slic* slic, int constantM){
	Mat img;
	src.convertTo(img, CV_64FC3);
	vector<double> center;
	slic->step = (int)sqrt((double)(src.rows * src.cols) / (double)superpixelNr);
	slic->constantM = constantM;

	for (int i = 0; i < img.rows; i++){
		vector<int> lb;
		vector<double> dst;

		for (int j = 0; j < img.cols; j++){

			lb.push_back(-1);
			dst.push_back(FLT_MAX);
		}

		slic->labels.push_back(lb);
		slic->distances.push_back(dst);
	}

	for (int i = slic->step / 2; i < img.rows - slic->step / 2; i += slic->step){
		for (int j = slic->step / 2; j < img.cols - slic->step / 2; j += slic->step){
			Point minim_point = findMinumum(img, Point(i, j));
			Vec3d color = img.at<Vec3d>(minim_point.x, minim_point.y);
			center.clear();

			center.push_back(color[0]);
			center.push_back(color[1]);
			center.push_back(color[2]);
			center.push_back((double)minim_point.x);
			center.push_back((double)minim_point.y);

			slic->clusters.push_back(center);
			slic->occurrencesCluster.push_back(0);
		}
	}
}

void clear(Slic* slic){
	slic->labels.clear();
	slic->distances.clear();
	slic->clusters.clear();
	slic->occurrencesCluster.clear();
	slic->step = 0;
	slic->constantM = 0;
}

double findDistance(Point position, Vec3d color, int cluster, Slic* slic){
	double spatialDistance = 0.0;
	double colorDistance = 0.0f;
	vector<double> clusterCenter = slic->clusters.at(cluster);

	colorDistance = sqrt(pow(clusterCenter[0] - color[0], 2.0) + pow(clusterCenter[1] - color[1], 2.0) + pow(clusterCenter[2] - color[2], 2.0));
	spatialDistance = sqrt(pow(clusterCenter[3] - (double)position.x, 2.0) + pow(clusterCenter[4] - (double)position.y, 2.0));

	colorDistance = colorDistance / slic->step;
	spatialDistance = spatialDistance / (double)slic->constantM;

	return sqrt(pow(colorDistance, 2.0) + pow(spatialDistance, 2.0));
}


void displayClusters(Mat src, Slic* slic){
	Mat dest = src.clone();
	std::cout << slic->clusters.size() << std::endl;

	for (int i = 0; i < slic->clusters.size(); i++){
		circle(dest, Point(slic->clusters.at(i).at(3), slic->clusters.at(i).at(4)), 1, Scalar(0, 0, 255));
	}

	imshow("Center", dest);
}

void generateSuperpixel(Mat src, int superpixelNr, int constantM, Slic* slic){
	clear(slic);
	initialize(src, superpixelNr, slic, constantM);
	displayClusters(src, slic);
}

vector<vector<int>> getLabels(Slic* slic){
	return slic->labels;
}

void createConnectivity(Mat* src){

}

float findLABSpaceValue(double t, double eps){
	 
	if (t > eps)
		return pow(t, 1.0 / 3.0);
	else
		return 7.787 * t + 16.0 / 116.0;
}

Vec3d BGR2LAB(Vec3b pixel){
	
	Vec3d LAB;
	double X, Y, Z;
	double eps1 = 0.008856;
	double eps2 = 903.3;
	double XR = 0.950456;
	double YR = 1.0f;
	double ZR = 1.088754;

	double B = (double)pixel[0] / 255.0;
	double G = (double)pixel[1] / 255.0;
	double R = (double)pixel[2] / 255.0;

	X = R * 0.412453 + G * 0.357580 + B * 0.180423;
	Y = R * 0.212671 + G * 0.715160 + B * 0.072169;
	Z = R * 0.019334 + G * 0.119193 + B * 0.950227;

	double xr = X / XR;
	double yr = Y / YR;
	double zr = Z / ZR;

	LAB[0] = (yr > eps1) ? (116.0 * (pow(yr, 1.0 / 3.0) - 16.0)) : (eps2 * yr);
	LAB[1] = 500.0 * (findLABSpaceValue(xr, eps1) - findLABSpaceValue(yr, eps1));
	LAB[2] = 200.0 * (findLABSpaceValue(yr, eps1) - findLABSpaceValue(zr, eps1));

	LAB[0] = LAB[0] * 255.0 / 100.0;
	LAB[1] = LAB[1] + 128.0;
	LAB[2] = LAB[2] + 128.0;
	return LAB;
}

Mat RGB2LABConversion(Mat src){

	Mat dest(src.rows, src.cols, CV_8UC3);
	int mina, minb, minl, maxa, maxb, maxl;
	mina = INT_MAX;
	minb = INT_MAX;
	minl = INT_MAX;
	maxa = INT_MIN;
	maxb = INT_MIN;
	maxl = INT_MIN;

	for (int i = 0; i < src.rows; i++){
		for (int j = 0; j < src.cols; j++){
			Vec3b pixel = src.at<Vec3b>(i, j);
			dest.at<Vec3b>(i, j) = BGR2LAB(src.at<Vec3b>(i, j));

			if (mina > pixel[1])
				mina = pixel[1];
			if (maxa < pixel[1])
				maxa = pixel[1];
			if (minl > pixel[0])
				minl = pixel[0];
			if (maxl < pixel[0])
				maxl = pixel[0];
			if (minb > pixel[2])
				minb = pixel[2];
			if (maxb < pixel[2])
				maxb = pixel[2];
		}
	}

	printf("LMin: %d\nLMax: %d\nAMin: %d\nAMax: %d\n BMin: %d\n BMax: %d\n", minl, maxl, mina, maxa, minb, maxb);
	return dest;
}

Mat colorByClusters(Mat src, Slic* slic){
	vector<Vec3b> clustersColors(slic->clusters.size());
	Mat dest(src.rows, src.cols, CV_8UC3);

	for (int i = 0; i < src.rows; i++){
		for (int j = 0; j < src.cols; j++){
			int label = slic->labels.at(i).at(j);
			Vec3b color = src.at<Vec3b>(i, j);
			clustersColors.at(label).val[0] += color.val[0];
			clustersColors.at(label).val[1] += color.val[1];
			clustersColors.at(label).val[2] += color.val[2];
		}
	}

	for (int i = 0; i < clustersColors.size(); i++){
		clustersColors.at(i).val[0] = clustersColors.at(i).val[0] / slic->occurrencesCluster.at(i);
		clustersColors.at(i).val[1] = clustersColors.at(i).val[1] / slic->occurrencesCluster.at(i);
		clustersColors.at(i).val[2] = clustersColors.at(i).val[2] / slic->occurrencesCluster.at(i);
	}

	for (int i = 0; i < src.rows; i++){
		for (int j = 0; j < src.cols; j++){
			dest.at<Vec3b>(i, j) = clustersColors.at(slic->labels.at(i).at(j));
		}
	}

	return dest;
}

void on_trackbar(int, void*){

	Slic slic;
	Mat src;

	src = imread(fname, CV_LOAD_IMAGE_COLOR);
	generateSuperpixel(src, k_slider + 1, m_slider + 1, &slic);
	std::cout << "Grid step: " << slic.step << std::endl;
	imshow("Image", src);
}

int main()
{
	int op;
	int k_slider_maxim;
	int m_slider_maxim;

	do
	{
		system("cls");
		destroyAllWindows();

		Vec3b pixel = BGR2LAB(Vec3b(0, 0, 255));

		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Resize image\n");
		printf(" 4 - Process video\n");
		printf(" 5 - Snap frame from live video\n");
		printf(" 6 - Mouse callback demo\n");
		printf(" 7 - RGB2LAB conversion\n");
		printf(" 8 - Cluster center\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testResize();
				break;
			case 4:
				testVideoSequence();
				break;
			case 5:
				testSnap();
				break;
			case 6:
				testMouseClick();
				break;

			case 7:
				while (openFileDlg(fname))
				{
					Mat src;
					src = imread(fname, CV_LOAD_IMAGE_COLOR);
					Mat dest(src.rows, src.cols, CV_8UC3);

					dest = RGB2LABConversion(src);
					Mat dest2(src.rows, src.cols, CV_8UC3);
					cvtColor(src, dest2, CV_BGR2Lab, 3);

					imshow("rgb", src);
					imshow("lab", dest);
					imshow("lab2", dest2);
					waitKey(0);
				}
				break;

			case 8:
				while (openFileDlg(fname))
				{
					namedWindow("TrackBar", 1);
					resizeWindow("TrackBar", 1000, 100);

					Mat src;
					src = imread(fname, CV_LOAD_IMAGE_COLOR);
					imshow("Image", src);
					imshow("Center", src);

					char TrackbarName[50];
					k_slider_maxim = (src.rows * src.cols) / 9 - 1;
					m_slider_maxim = 500;
					sprintf(TrackbarName, "K %d", k_slider_maxim);
					createTrackbar(TrackbarName, "TrackBar", &k_slider, k_slider_maxim, on_trackbar);
					sprintf(TrackbarName, "M %d", m_slider_maxim);
					createTrackbar(TrackbarName, "TrackBar", &m_slider, m_slider_maxim, on_trackbar);
					waitKey(0);
					destroyAllWindows();
				}
				break;
		}
	}
	while (op!=0);
	return 0;
}