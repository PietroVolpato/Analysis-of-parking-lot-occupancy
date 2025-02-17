#include "ParkingSpaceDetector.h"

using namespace cv;
using namespace std;

vector<Mat> ParkingSpaceDetector::loadImages(const int sequence) {
    vector<String> fileNames;
    if (sequence == 0)
        glob("../data/sequence0/frames", fileNames);
    else if (sequence == 1)
        glob("./data/sequence1/frames", fileNames);
    else if (sequence == 2)
        glob("./data/sequence2/frames", fileNames);
    else if (sequence == 3)
        glob("./data/sequence3/frames", fileNames);
    else if (sequence == 4)
        glob("./data/sequence4/frames", fileNames);
    else if (sequence == 5)
        glob("./data/sequence5/frames", fileNames);

    vector<Mat> imgs;
    for (const auto& file : fileNames) {
        Mat img = imread(file);
        if (img.empty()) {
            cerr << "Error loading image: " << file << endl;
            continue;
        }
        imgs.push_back(img);
    }
    return imgs;
}

vector<LineParams> ParkingSpaceDetector::computeLineParams(const vector<Vec4i>& lines) {
    vector<LineParams> params;
    for (const auto& line : lines) {
        int x1 = line[0], y1 = line[1], x2 = line[2], y2 = line[3];
        double a = y1 - y2;
        double b = x2 - x1;
        double c = x1 * y2 - x2 * y1;
        // angle in degrees
        double theta = atan2(b, a) * 180 / CV_PI;
        theta = std::fmod(theta + 90, 180) - 90;
        double length = sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
        params.push_back({theta, line, length});
    }
    return params;
}

Mat ParkingSpaceDetector::applyRoi(const Mat& img) {
    Mat mask = Mat::zeros(img.size(), img.type());
    vector<vector<Point>> roi;
    Point pts[4] = {
        Point(0, 0),
        Point(img.cols * 0.15, 0),
        Point(img.cols * 0.47 , img.rows),
        Point(0, img.rows),
    };
    roi.push_back(vector<Point>(pts, pts + 4));
    fillPoly(mask, roi, Scalar(255));

    bitwise_not(mask, mask);

    Mat masked;
    bitwise_and(img, mask, masked);

    roi.clear();
    mask = Mat::zeros(img.size(), img.type());

    Point pts2[4] = {
        Point(img.cols * 0.65, 0),
        Point(img.cols, 0),
        Point(img.cols, img.rows * 0.4),
        Point(img.cols * 0.65, 0)
    };
    roi.push_back(vector<Point>(pts2, pts2 + 4));
    fillPoly(mask, roi, Scalar(255));

    bitwise_not(mask, mask);
    bitwise_and(masked, mask, masked);
    return masked;
}

Mat ParkingSpaceDetector::equalization(const Mat& img) {
    Ptr<CLAHE> clahe = cv::createCLAHE(2.0, Size(8, 8));
    Mat out;
    clahe->apply(img, out);

    return out;
}

float ParkingSpaceDetector::distance(const Vec4i& line1, const Vec4i& line2) {
    float mx1 = (line1[0] + line1[2]) / 2.0;
    float my1 = (line1[1] + line1[3]) / 2.0;
    float mx2 = (line2[0] + line2[2]) / 2.0;
    float my2 = (line2[1] + line2[3]) / 2.0;

    return hypot(mx2 - mx1, my2 - my1);
}

Vec4i ParkingSpaceDetector::mergeLines(const Vec4i line1, const Vec4i& line2) {
    return Vec4i(min(line1[0], line2[0]), min(line1[1], line2[1]),
                 max(line1[2], line2[2]), max(line1[3], line2[3]));

}

bool ParkingSpaceDetector::isParallel(const double theta1, const double theta2) {
    double tollerance = 5;
    double diff = abs(abs(theta1) - abs(theta2));
    diff = min(diff, 180 - diff);

    return diff <= tollerance;
}

vector<Vec4i> ParkingSpaceDetector::mergeLines(const vector<LineParams>& lines) {
    vector<Vec4i> mergedLines;
    vector<bool> used(lines.size(), false);

    for (size_t i = 0; i < lines.size(); ++i) {
        if (used[i]) continue;
        Vec4i merged = lines[i].endpoints;
        for (size_t j = i + 1; j < lines.size(); ++j) {
            if (used[j]) continue;
            if (distance(lines[i].endpoints, lines[j].endpoints) < 10 && isParallel(lines[i].theta, lines[j].theta)) {
                merged = mergeLines(merged, lines[j].endpoints);
                used[j] = true;
            }
        }
        mergedLines.push_back(merged);
    }

    return mergedLines;
}

Mat ParkingSpaceDetector::preprocessImage(const Mat& img) {
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    Mat equalized = equalization(gray);
    // Mat blurred;
    // GaussianBlur(equalized, blurred, Size(3, 3), 0);
    // bilateralFilter(gray, blurred, 3, 50, 50);
    Mat thresh;
    adaptiveThreshold(gray, thresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 9, 15);
    // Mat closed;
    // Mat kernel = getStructuringElement(MORPH_RECT, Size(7, 7));
    // morphologyEx(thresh, closed, MORPH_CLOSE, kernel);
    // Mat thinned;
    // kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    // erode(closed, thinned, kernel);
    Mat thresh_masked = applyRoi(thresh);

    Mat sobelX, sobelY;
    Sobel(equalized, sobelX, CV_32F, 1, 0);
    Sobel(equalized, sobelY, CV_32F, 0, 1);
    Mat sobel;
    magnitude(sobelX, sobelY, sobel);
    normalize(sobel, sobel, 0, 255, NORM_MINMAX);
    sobel.convertTo(sobel, CV_8U);
    Mat sobel_masked = applyRoi(sobel);

    Mat out;
    out = thresh_masked + sobel_masked;

    // morphologyEx(out, out, MORPH_CLOSE, kernel);

    return thresh_masked;
}

Mat ParkingSpaceDetector::detectEdges(const Mat& img, int threshold1, int threshold2) {
    Mat edges;
    Canny(img, edges, threshold1, threshold2);
    return edges;
}

void ParkingSpaceDetector::showImage(const Mat& img) {
    imshow("Image", img);
    waitKey(0);
}

vector<Vec4i> ParkingSpaceDetector::detectLines(const Mat& img, int threshold, double minLineLength, double maxLineGap) {
    vector<Vec4i> lines;
    HoughLinesP(img, lines, 1, CV_PI/180, threshold, minLineLength, maxLineGap);
    return lines;
}

void ParkingSpaceDetector::drawLines(Mat& img, const vector<LineParams>& lines) {
    for (const auto& line : lines) {
        Point p1(line.endpoints[0], line.endpoints[1]);
        Point p2(line.endpoints[2], line.endpoints[3]);
        cv::line(img, p1, p2, Scalar(0, 0, 255), 2);
    }
}

vector<LineParams> ParkingSpaceDetector::filterLines(vector<LineParams>& lines) {
    vector<LineParams> filteredLines;
    double angleTarget = 75;
    double tollerance = 10;
    for (const auto& line: lines) {
        double angle = line.theta;
        if (abs(abs(angle) - angleTarget) <= tollerance) {
            filteredLines.push_back(line);
        }
    }

    vector<Vec4i> mergedLines = mergeLines(filteredLines);
    filteredLines = computeLineParams(mergedLines);

    return filteredLines;
}

vector<vector<LineParams>> ParkingSpaceDetector::clusterLinesByTheta(const vector<LineParams>& lines, double thetaThreshold) {
    vector<vector<LineParams>> clusters;
    
    return clusters;
}

vector<RotatedRect> ParkingSpaceDetector::detectParkingSpaces(const vector<LineParams>& lineParams) {
    vector<RotatedRect> parkingSpaces;
    
    return parkingSpaces;
}

Mat ParkingSpaceDetector::drawParkingSpaces(const Mat& img, const vector<RotatedRect>& parkingSpaces) {
    Mat out = img.clone();
    for (const auto& space : parkingSpaces) {
        Point2f vertices[4];
        space.points(vertices);
        for (int i = 0; i < 4; ++i) {
            line(out, vertices[i], vertices[(i+1)%4], Scalar(0, 255, 0), 2);
        }
    }
    return out;
}