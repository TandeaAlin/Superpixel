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
	double min = DBL_MAX;
	Point min_point = Point(center.x, center.y);

	for (int i = -1; i <= 1; i++){
		for (int j = -1; j <= 1; j++){
			Vec3b p1 = Vec3b(0, 0, 0);
			Vec3b p2 = Vec3b(0, 0, 0);
			Vec3b p3 = Vec3b(0, 0, 0);

			if ((center.x + i + 1 >= 0) && (center.x + i + 1 < img.rows) && (center.y + j >= 0) && (center.y + j < img.cols))
				p1 = img.at<Vec3b>(center.x + i + 1, center.y + j);
			if ((center.x + i >= 0) && (center.x + i < img.rows) && (center.y + j + 1 >= 0) && (center.y + j + 1< img.cols))
				p2 = img.at<Vec3b>(center.x + i, center.y + j + 1);
			if ((center.x + i >= 0) && (center.x + i < img.rows) && (center.y + j >= 0) && (center.y + j < img.cols))
				p3 = img.at<Vec3b>(center.x + i, center.y + j);
		
			double c1 = sqrt(pow((double)(p1[0] - p3[0]), 2.0)) + sqrt(pow((double)(p2[0] - p3[0]), 2.0));
			double c2 = sqrt(pow((double)(p1[1] - p3[1]), 2.0)) + sqrt(pow((double)(p2[1] - p3[1]), 2.0));
			double c3 = sqrt(pow((double)(p1[2] - p3[2]), 2.0)) + sqrt(pow((double)(p2[2] - p3[2]), 2.0));

			if ((c1 + c2 + c3) < min){
				min = fabs((double)(p1[0] - p3[0])) + fabs((double)(p2[0] - p3[0])) + fabs((double)(p1[1] - p3[1])) + fabs((double)(p2[1] - p3[1])) + 
					fabs((double)(p1[2] - p3[2])) + fabs((double)(p2[2] - p3[2]));
				min_point.x = center.x + i;
				min_point.y = center.y + j;
			}

		}
	}

	return min_point;
}

void initialize(Mat img, int superpixelNr, Slic* slic, int constantM){
	vector<double> center;
	slic->step = (int)sqrt((double)(img.rows * img.cols) / (double)superpixelNr);
	slic->constantM = constantM;

	for (int i = 0; i < img.rows; i++){
		vector<int> lb;
		vector<double> dst;

		for (int j = 0; j < img.cols; j++){

			lb.push_back(-1);
			dst.push_back(DBL_MAX);
		}

		slic->labels.push_back(lb);
		slic->distances.push_back(dst);
	}

	for (int i = slic->step / 2; i < img.rows - slic->step / 2; i += slic->step){
		for (int j = slic->step / 2; j < img.cols - slic->step / 2; j += slic->step){
			Point minim_point = findMinumum(img, Point(i, j));
			Vec3b color = img.at<Vec3b>(minim_point.x, minim_point.y);
			center.clear();

			center.push_back((double)color[0]);
			center.push_back((double)color[1]);
			center.push_back((double)color[2]);
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

void initializeDistances(Slic* slic){
	for (int i = 0; i < slic->distances.size(); i++)
		for (int j = 0; j < slic->distances.at(i).size(); j++)
			slic->distances.at(i).at(j) = DBL_MAX;
}

double findDistance(Point position, Vec3b color, int cluster, Slic* slic){
	double spatialDistance = 0.0;
	double colorDistance = 0.0;
	vector<double> clusterCenter = slic->clusters.at(cluster);

	colorDistance = sqrt(pow(clusterCenter[0] - (double)color[0], 2.0) + pow(clusterCenter[1] - (double)color[1], 2.0) + pow(clusterCenter[2] - (double)color[2], 2.0));
	spatialDistance = sqrt(pow(clusterCenter[3] - (double)position.x, 2.0) + pow(clusterCenter[4] - (double)position.y, 2.0));

	colorDistance = colorDistance / slic->step;
	spatialDistance = spatialDistance / (double)slic->constantM;
	return sqrt(pow(colorDistance, 2.0) + pow(spatialDistance, 2.0));
}


void displayClusters(Mat src, Slic* slic){
	Mat dest = src.clone();

	for (int i = 0; i < slic->clusters.size(); i++)
		circle(dest, Point(slic->clusters.at(i).at(3), slic->clusters.at(i).at(4)), 1, Scalar(0, 0, 255));

	imshow("Center", dest);
}

void displayContours(Mat img, Slic* slic){
	Mat dest = img.clone();
	int dx[] = { -1, -1, -1, 0, 0, 1, 1, 1 };
	int dy[] = { -1, 0, 1, -1, 1, -1, 0, 1 };
	vector<Point> contourPoints;
	vector<vector<bool>> isContourPixel;

	for (int i = 0; i < img.rows; i++){
		vector<bool> contourLine;
		for (int j = 0; j < img.cols; j++)
			contourLine.push_back(false);
		isContourPixel.push_back(contourLine);
	}

	for (int i = 0; i < img.rows; i++){
		for (int j = 0; j < img.cols; j++){
			int pixelNr = 0;

			for (int k = 0; k < 8; k++){
				int line = i + dx[i];
				int col = j + dx[j];
				if ((line >= 0) && (line < img.rows) && (col >= 0) && (col < img.cols))
					if (!isContourPixel.at(i).at(j) && slic->labels.at(i).at(j) != slic->labels.at(line).at(col))
						pixelNr += 1;
			}

			if (pixelNr > 1){
				isContourPixel.at(i).at(j) = true;
				contourPoints.push_back(Point(i, j));
			}
		}
	}

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			dest.at<Vec3b>(i, j) = Vec3b(255, 255, 255);

	imshow("Superpixel", dest);
}

void colorByClusters(Mat src, Slic* slic){
	vector<Vec3b> clustersColors(slic->clusters.size(), Vec3b(0, 0, 0));
	Mat dest = src.clone();

	for (int i = 0; i < src.rows; i++){
		for (int j = 0; j < src.cols; j++){
			int label = slic->labels.at(i).at(j);
			if (label > 0){
				Vec3b color = src.at<Vec3b>(i, j);
				clustersColors.at(label).val[0] += color.val[0];
				clustersColors.at(label).val[1] += color.val[1];
				clustersColors.at(label).val[2] += color.val[2];
			}
		}
	}

	for (int i = 0; i < clustersColors.size(); i++){
		if (slic->occurrencesCluster.at(i) != 0){
			clustersColors.at(i).val[0] = clustersColors.at(i).val[0] / slic->occurrencesCluster.at(i);
			clustersColors.at(i).val[1] = clustersColors.at(i).val[1] / slic->occurrencesCluster.at(i);
			clustersColors.at(i).val[2] = clustersColors.at(i).val[2] / slic->occurrencesCluster.at(i);
		}
	}

	for (int i = 0; i < src.rows; i++){
		for (int j = 0; j < src.cols; j++){
			if (slic->labels.at(i).at(j) > 0)
				dest.at<Vec3b>(i, j) = clustersColors.at(slic->labels.at(i).at(j));
		}
	}

	displayContours(dest, slic);
}

void generateSuperpixel(Mat src, int superpixelNr, int constantM, Slic* slic){
	Mat img(src.rows, src.cols, CV_8UC3);
	cvtColor(src, img, CV_BGR2Lab, 3);
	clear(slic);
	initialize(img, superpixelNr, slic, constantM);

	for (int k = 0; k < slic->clusters.size(); k++){
		for (int i = -slic->step; i < slic->step; i++){
			for (int j = -slic->step; j < slic->step; j++){
				int row = slic->clusters.at(k).at(3);
				int col = slic->clusters.at(k).at(4);

				if ((row >= 0) && (row < img.rows) && (col >= 0) && (col < img.cols)){
					Vec3b color = src.at<Vec3b>(row, col);
					double distance = findDistance(Point(row, col), color, k, slic);
					if (distance < slic->distances.at(row).at(col)){
						slic->distances.at(row).at(col) = distance;
						slic->labels.at(row).at(col) = k;
					}
				}
			}
		}
	}

	colorByClusters(src, slic);
}

void createConnectivity(Mat* src){

}

double findLABSpaceValue(double t, double eps){
	 
	if (t > eps)
		return pow(t, 1.0 / 3.0);
	else
		return 7.787 * t + 16.0 / 116.0;
}

Vec3b BGR2LAB(Vec3b pixel){
	
	Vec3b LAB;
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

	for (int i = 0; i < src.rows; i++){
		for (int j = 0; j < src.cols; j++){
			dest.at<Vec3b>(i, j) = BGR2LAB(src.at<Vec3b>(i, j));
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

void on_trackbar_video(int, void*) {
	Slic slic;
	Mat frame;// here we get the input from the webcam

	VideoCapture cap(0);

	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	//video res
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	char c;
	int frameNum = -1;
	int frameCount = 0;

	

	Mat resize(capS.height , capS.height , CV_8UC3);

	std::cout << "width : " << capS.width << "height : " << capS.height<< std::endl;

	int resizeFactor = capS.width - capS.height;

	for (;;)
	{
		cap >> frame; // get a new frame from camera

		for (int i = 0; i < capS.height ; i++) {
			for (int j = 0 ; j < capS.width ; j++) {
				resize.at<Vec3b>(i, j) = frame.at<Vec3b>(i, j + resizeFactor);
			}
		}


		generateSuperpixel(resize, k_slider + 1, m_slider + 1, &slic);

		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;

		imshow("Image", resize);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
	}


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
		printf(" 1 - RGB2LAB conversion\n");
		printf(" 2 - Cluster center\n");
		printf(" 3 - Cluster center on video\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
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

			case 2:
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
			case 3:

				Mat frame;

				VideoCapture cap(0);

				cap >> frame; // get 1 frame from the camera;

				k_slider_maxim = (frame.rows * frame.cols) / 9 - 1;
				m_slider_maxim = 500;

				namedWindow("TrackBar", 1);
				resizeWindow("TrackBar", 1000, 100);


				char TrackbarNameVideo[50];

				sprintf(TrackbarNameVideo, "K %d", k_slider_maxim);
				createTrackbar(TrackbarNameVideo, "TrackBar", &k_slider, k_slider_maxim, on_trackbar_video);
				sprintf(TrackbarNameVideo, "M %d", m_slider_maxim);
				createTrackbar(TrackbarNameVideo, "TrackBar", &m_slider, m_slider_maxim, on_trackbar_video);

			


				waitKey(0);
				destroyAllWindows();

				break;
		}
	}
	while (op!=0);
	return 0;
}