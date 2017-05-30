#include "stdafx.h"
#include "common.h"
#include <vector>

typedef struct{
	int step;
	double constantM;
	vector<vector<int>> labels;
	vector<vector<double>> distances;
	vector<double> occurrencesCluster;
	vector<vector<double>> clusters;
}Slic;

int k_slider;
int m_slider;
char fname[MAX_PATH];

Vec3b BGR2HSV(Vec3b pixel){
	double r = (double)pixel[2] / 255.0;
	double g = (double)pixel[1] / 255.0;
	double b = (double)pixel[0] / 255.0;

	double M = max(max(r, g), b);
	double m = min(min(r, g), b);
	double C = M - m;

	double V = M;
	double S = 0.0;
	double H = 0.0;

	if (C)
		S = C / V;

	if (C){
		if (M == r)
			H = 60.0 * (g - b) / C;
		if (M == g)
			H = 120.0 + 60.0 * (b - r) / C;
		if (M == b)
			H = 240.0 + 60.0 * (r - g) / C;
	}

	if (H < 0.0)
		H = H + 360.0;

	H = H * 255.0 / 360.0;
	S = S * 255.0;
	V = V * 255.0;

	return Vec3b((uchar)H, (uchar)S, (uchar)V);
}

Mat BGR2HSVConversion(Mat img){
	Mat dest(img.rows, img.cols, CV_8UC3);

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			dest.at<Vec3b>(i, j) = BGR2HSV(img.at<Vec3b>(i, j));

	return dest;
}

Point findMinumum(Mat img, Point center){
	double min = DBL_MAX;
	Point min_point = Point(center.x, center.y);

	for (int i = -1; i <= 1; i++)
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

			double c2 = sqrt(pow((double)(p1[1] - p3[1]), 2.0)) + sqrt(pow((double)(p2[1] - p3[1]), 2.0));
			double c3 = sqrt(pow((double)(p1[2] - p3[2]), 2.0)) + sqrt(pow((double)(p2[2] - p3[2]), 2.0));

			if ((c2 + c3) < min){
				min = fabs((double)(p1[1] - p3[1])) + fabs((double)(p2[1] - p3[1])) + fabs((double)(p1[2] - p3[2])) + fabs((double)(p2[2] - p3[2]));
				min_point.x = center.x + i;
				min_point.y = center.y + j;
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

	for (int i = slic->step / 2; i < img.rows; i += slic->step)
		for (int j = slic->step / 2; j < img.cols; j += slic->step){
			Point minim_point = findMinumum(img, Point(i, j));
			Vec3b color = img.at<Vec3b>(minim_point.x, minim_point.y);
			center.clear();

			center.push_back((double)color[0]);
			center.push_back((double)color[1]);
			center.push_back((double)color[2]);
			center.push_back(minim_point.x);
			center.push_back(minim_point.y);

			slic->clusters.push_back(center);
			slic->occurrencesCluster.push_back(0.0);
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

void updateClustersOccurences(Slic* slic, int maximLabel){
	slic->occurrencesCluster.clear();

	for (int i = 0; i < maximLabel; i++)
		slic->occurrencesCluster.push_back(0.0);

	for (int i = 0; i < slic->labels.size(); i++)
		for (int j = 0; j < slic->labels.at(i).size(); j++)
			slic->occurrencesCluster.at(slic->labels.at(i).at(j)) += 1.0;
}

void CleanClusters(Slic* slic){
	for (int k = 0; k < slic->clusters.size(); k++){
		slic->clusters.at(k).at(0) = 0.0;
		slic->clusters.at(k).at(1) = 0.0;
		slic->clusters.at(k).at(2) = 0.0;
		slic->clusters.at(k).at(3) = 0.0;
		slic->clusters.at(k).at(4) = 0.0;
		slic->occurrencesCluster.at(k) = 0.0;
	}
}

double findDistance(Point position, Vec3b color, int cluster, Slic* slic){
	double spatialDistance = 0.0;
	double colorDistance = 0.0;
	vector<double> clusterCenter = slic->clusters.at(cluster);

	colorDistance = sqrt(pow(clusterCenter[0] - (double)color[0], 2.0) + pow(clusterCenter[1] - (double)color[1], 2.0) + pow(clusterCenter[2] - (double)color[2], 2.0));
	spatialDistance = sqrt(pow(clusterCenter[3] - (double)position.x, 2.0) + pow(clusterCenter[4] - (double)position.y, 2.0));

	colorDistance = colorDistance / slic->step;
	spatialDistance = spatialDistance / slic->constantM;
	return sqrt(pow(colorDistance, 2.0) + pow(spatialDistance, 2.0));
}


void displayClusters(Mat src, Slic* slic){
	Mat dest = src.clone();

	for (int i = 0; i < slic->clusters.size(); i++)
		circle(dest, Point(slic->clusters.at(i).at(3), slic->clusters.at(i).at(4)), 1, Scalar(0, 0, 255));

	imshow("Center", dest);
}

void colorByClusters(Mat src, Slic* slic){
	vector<vector<long int>> clustersColors;
	Mat dest(src.rows, src.cols, CV_8UC3);

	vector<long int> initialize;
	initialize.push_back(0);
	initialize.push_back(0);
	initialize.push_back(0);
	for (int i = 0; i < slic->occurrencesCluster.size(); i++)
		clustersColors.push_back(initialize);

	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++){
			int label = slic->labels.at(i).at(j);
			if (label >= 0){
				Vec3b color = src.at<Vec3b>(i, j);
				clustersColors.at(label).at(0) += color[0];
				clustersColors.at(label).at(1) += color[1];
				clustersColors.at(label).at(2) += color[2];
			}
		}

	for (int i = 0; i < slic->occurrencesCluster.size(); i++)
		if (slic->occurrencesCluster.at(i) != 0.0){
			clustersColors.at(i).at(0) = clustersColors.at(i).at(0) / slic->occurrencesCluster.at(i);
			clustersColors.at(i).at(1) = clustersColors.at(i).at(1) / slic->occurrencesCluster.at(i);
			clustersColors.at(i).at(2) = clustersColors.at(i).at(2) / slic->occurrencesCluster.at(i);
		}

	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
			if (slic->labels.at(i).at(j) >= 0){
				initialize = clustersColors.at(slic->labels.at(i).at(j));
				dest.at<Vec3b>(i, j) = Vec3b(initialize.at(0), initialize.at(1), initialize.at(2));
			}

	imshow("Superpixel", dest);
}

void display_contours(Mat src, Slic* slic) {
	const int dx[] = { -1, -1, -1, 0, 0, 1, 1, 1 };
	const int dy[] = { -1, 0, 1, -1, 1, -1, 0, 1 };
	Mat dest = src.clone();

	vector<Point> contourPoints;
	vector<vector<bool>> isselected;
	for (int i = 0; i < src.rows; i++) {
		vector<bool> neg;
		for (int j = 0; j < src.cols; j++)
			neg.push_back(false);

		isselected.push_back(neg);
	}

	for (int i = 0; i < src.rows; i++)
		for (int j = 4; j < src.cols; j++) {
			int pixelNumber = 0;

			for (int k = 0; k < 8; k++) {
				int x = i + dx[k], y = j + dy[k];

				if (x >= 0 && x < src.rows && y >= 0 && y < src.cols)
					if (isselected.at(x).at(y) == false && slic->labels.at(i).at(j) != slic->labels.at(x).at(y)) {
						pixelNumber += 1;
					}
			}

			if (pixelNumber > 1) {
				contourPoints.push_back(Point(i, j));
				isselected.at(i).at(j) = true;
			}
		}

	for (int i = 0; i < contourPoints.size(); i++) {
		Point pixel = contourPoints.at(i);
		dest.at<Vec3b>(pixel.x, pixel.y) = Vec3b(0, 0, 0);
	}

	imshow("Contour", dest);
}

void create_connectivity(int rows, int cols, Slic* slic) {
	int label = 0, adjLabel = 0;
	const int lims = (cols * rows) / (int)((slic->clusters).size());

	const int dx[4] = { -1, 0, 1, 0 };
	const int dy[4] = { 0, -1, 0, 1 };

	// Initialize new cluster matrix
	vector<vector<int>> new_clusters;
	for (int i = 0; i < cols; i++) {
		vector<int> nc;
		for (int j = 0; j < rows; j++) {
			nc.push_back(-1);
		}
		new_clusters.push_back(nc);
	}

	for (int i = 0; i < cols; i++) {
		for (int j = 0; j < rows; j++) {

			if (new_clusters[i][j] == -1) {
				vector<Point> elements;
				elements.push_back(Point(i, j));

				//Find possible adjacent label
				for (int k = 0; k < 4; k++) {
					int x = elements[elements.size() - 1].x + dx[k];
					int y = elements[elements.size() - 1].y + dy[k];

					if (x >= 0 && x < cols && y >= 0 && y < rows) {
						if (new_clusters[x][y] >= 0) {
							adjLabel = new_clusters[x][y];
						}
					}
				}

				int count = 1;
				for (int c = 0; c < count; c++)
					for (int k = 0; k < 4; k++) {
						int x = elements[c].x + dx[k];
						int y = elements[c].y + dy[k];

						if (x >= 0 && x < cols && y >= 0 && y < rows)
							if (new_clusters[x][y] == -1 && slic->labels[i][j] == slic->labels[x][y]) {
								elements.push_back(Point(x, y));
								new_clusters[x][y] = label;
								count += 1;
							}
					}

				/* Using the earlier found adjacent label if a segment size is
				smaller than a limit. */
				if (count <= lims >> 2) {
					for (int c = 0; c < count; c++){
						new_clusters[elements[c].x][elements[c].y] = adjLabel;
					}
					label -= 1;
				}
				label += 1;
			}
		}
	}

	slic->labels = new_clusters;
	updateClustersOccurences(slic, label);
}

void generateSuperpixel(Mat src, int superpixelNr, Slic* slic){
	Mat dest(src.rows, src.cols, CV_8UC3);
	Mat img(src.rows, src.cols, CV_8UC3);

	clear(slic);
	img = BGR2HSVConversion(src);
	initialize(img, superpixelNr, slic, 10.0);

	for (int iteration = 0; iteration < 10; iteration++){

		initializeDistances(slic);

		for (int k = 0; k < slic->clusters.size(); k++)
			for (int i = -slic->step; i < slic->step; i++)
				for (int j = -slic->step; j < slic->step; j++){

					int row = slic->clusters.at(k).at(3) + i;
					int col = slic->clusters.at(k).at(4) + j;

					if ((row >= 0) && (row < img.rows) && (col >= 0) && (col < img.cols)){
						Vec3b color = img.at<Vec3b>(row, col);
						double distance = findDistance(Point(row, col), color, k, slic);
						if (distance < slic->distances.at(row).at(col)){
							slic->distances.at(row).at(col) = distance;
							slic->labels.at(row).at(col) = k;
						}
					}
				}

		CleanClusters(slic);

		for (int i = 0; i < img.rows; i++)
			for (int j = 0; j < img.cols; j++){
				int index = slic->labels.at(i).at(j);

				if (index >= 0){
					Vec3b color = img.at<Vec3b>(i, j);
					slic->clusters.at(index).at(0) += color[0];
					slic->clusters.at(index).at(1) += color[1];
					slic->clusters.at(index).at(2) += color[2];
					slic->clusters.at(index).at(3) += i;
					slic->clusters.at(index).at(4) += j;
					slic->occurrencesCluster.at(index) += 1.0;
				}
			}

		for (int k = 0; k < slic->clusters.size(); k++){
			slic->clusters.at(k).at(0) /= slic->occurrencesCluster.at(k);
			slic->clusters.at(k).at(1) /= slic->occurrencesCluster.at(k);
			slic->clusters.at(k).at(2) /= slic->occurrencesCluster.at(k);
			slic->clusters.at(k).at(3) /= slic->occurrencesCluster.at(k);
			slic->clusters.at(k).at(4) /= slic->occurrencesCluster.at(k);
		}
	}
}

void on_trackbar(int, void*){

	Slic slic;
	Mat src;
	src = imread(fname, CV_LOAD_IMAGE_COLOR);
	generateSuperpixel(src, k_slider + 1, &slic);
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

	Mat resize(capS.height, capS.height, CV_8UC3);

	std::cout << "width : " << capS.width << "height : " << capS.height << std::endl;

	int resizeFactor = capS.width - capS.height;

	for (;;)
	{
		cap >> frame; // get a new frame from camera

		for (int i = 0; i < capS.height; i++) {
			for (int j = 0; j < capS.width; j++) {
				resize.at<Vec3b>(i, j) = frame.at<Vec3b>(i, j + resizeFactor);
			}
		}


		generateSuperpixel(resize, k_slider + 1, &slic);

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
	char key;
	int asciiValue;
	Slic slic;

	do
	{
		system("cls");
		destroyAllWindows();

		printf("Menu:\n");
		printf(" 1 - Cluster center\n");
		printf(" 2 - Cluster center on video\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);

		switch (op)
		{
		case 1:
			printf("Number superpixel: ");
			scanf("%d", &k_slider);

			while (openFileDlg(fname))
			{
				Mat src;
				src = imread(fname, CV_LOAD_IMAGE_COLOR);
				int dim = (src.rows > src.cols) ? src.rows : src.cols;
				Mat img(dim, dim, CV_8UC3);
				resize(src, img, Size(dim, dim));

				double t = (double)getTickCount();

				generateSuperpixel(img, k_slider, &slic);
				imshow("Image", img);
				create_connectivity(img.rows, img.cols, &slic);
				colorByClusters(img, &slic);
				display_contours(img, &slic);

				t = ((double)getTickCount() - t) / getTickFrequency();
				printf("Time = %.3f [ms]\n", t * 1000);

				waitKey(0);
				destroyAllWindows();
			}
			break;

		case 2:

			Mat frame;

			VideoCapture cap(0);

			cap >> frame; // get 1 frame from the camera;

			k_slider_maxim = (frame.rows * frame.cols) / 9 - 1;
			namedWindow("TrackBar", 1);
			resizeWindow("TrackBar", 1000, 50);


			char TrackbarNameVideo[50];

			sprintf(TrackbarNameVideo, "K %d", k_slider_maxim);
			createTrackbar(TrackbarNameVideo, "TrackBar", &k_slider, k_slider_maxim, on_trackbar_video);

			waitKey(0);
			destroyAllWindows();

			break;
		}
	} while (op != 0);

	clear(&slic);
	return 0;
}