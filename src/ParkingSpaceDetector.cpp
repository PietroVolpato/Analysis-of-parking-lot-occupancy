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
        double rho = -c / sqrt(a * a + b * b);
        double length = sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
        params.push_back({rho, theta, line, length});
    }
    return params;
}

Mat ParkingSpaceDetector::detectEdges(const Mat& img) {
    Mat edges;
    Canny(img, edges, 50, 200);
    return edges;
}

Mat ParkingSpaceDetector::applyRoi(const Mat& img) {
    Mat mask = Mat::zeros(img.size(), img.type());
    vector<vector<Point>> roi;
    Point pts[4] = {
        Point(0, 0),
        Point(img.cols * 0.17, 0),
        Point(img.cols * 0.45 , img.rows),
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

Mat ParkingSpaceDetector::preprocessImage(const Mat& img) {
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    Mat blurred;
    // GaussianBlur(gray, blurred, Size(3, 3), 0);
    // bilateralFilter(gray, blurred, 3, 50, 50);
    Mat thresh;
    adaptiveThreshold(gray, thresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 7, 9);
    Mat closed;
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(thresh, closed, MORPH_CLOSE, kernel);
    Mat thinned;
    kernel = getStructuringElement(MORPH_CROSS, Size(3, 3));
    erode(closed, thinned, kernel);
    Mat masked = applyRoi(thinned);
    return masked;
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
    double angle_tolerance = 10.0;
    for (const auto& line : lines) {
        double angle = line.theta;
        double length = line.length;
        if (length > 50 && fabs(angle - 90) < angle_tolerance) {
            filteredLines.push_back(line);
        }
    }
    return filteredLines;
}

vector<vector<LineParams>> ParkingSpaceDetector::clusterLinesByTheta(const vector<LineParams>& lines, double thetaThreshold) {
    vector<vector<LineParams>> clusters;
    if (lines.empty()) return clusters;

    vector<LineParams> sortedLines = lines;
    sort(sortedLines.begin(), sortedLines.end(), [](const LineParams& a, const LineParams& b) {
        return a.theta < b.theta;
    });

    vector<LineParams> currentCluster = {sortedLines[0]};
    for (size_t i = 1; i < sortedLines.size(); ++i) {
        double deltaTheta = abs(sortedLines[i].theta - currentCluster.back().theta);
        // Adjust for angles wrapping around (if necessary)
        if (deltaTheta > 180) deltaTheta = 360 - deltaTheta;
        if (deltaTheta <= thetaThreshold) {
            currentCluster.push_back(sortedLines[i]);
        } else {
            clusters.push_back(currentCluster);
            currentCluster = {sortedLines[i]};
        }
    }
    if (!currentCluster.empty())
        clusters.push_back(currentCluster);
    return clusters;
}

vector<RotatedRect> ParkingSpaceDetector::detectParkingSpaces(const vector<LineParams>& lineParams) {
    double thetaThreshold = 5 * CV_PI / 180;
    vector<vector<LineParams>> clusters = clusterLinesByTheta(lineParams, thetaThreshold);

    vector<RotatedRect> parkingSpaces;
    for (const auto& cluster : clusters) {
        if (cluster.size() < 2) continue;

        vector<LineParams> sortedCluster = cluster;
        sort(sortedCluster.begin(), sortedCluster.end(), [](const LineParams& a, const LineParams& b) {
            return a.rho < b.rho;
        });

        vector<double> diffs;
        for (size_t i = 1; i < sortedCluster.size(); ++i) {
            diffs.push_back(sortedCluster[i].rho - sortedCluster[i-1].rho);
        }
        if (diffs.empty()) continue;

        sort(diffs.begin(), diffs.end());
        double medianDiff = diffs[diffs.size() / 2];
        double tolerance = 0.2 * medianDiff;

        for (size_t i = 0; i < sortedCluster.size() - 1; ++i) {
            double diff = sortedCluster[i+1].rho - sortedCluster[i].rho;
            if (abs(diff - medianDiff) <= tolerance) {
                const LineParams& line1 = sortedCluster[i];
                const LineParams& line2 = sortedCluster[i+1];

                Vec4i l1 = line1.endpoints;
                Vec4i l2 = line2.endpoints;
                Point2f mid1((l1[0] + l1[2]) / 2.0, (l1[1] + l1[3]) / 2.0);
                Point2f mid2((l2[0] + l2[2]) / 2.0, (l2[1] + l2[3]) / 2.0);
                Point2f center((mid1.x + mid2.x) / 2.0, (mid1.y + mid2.y) / 2.0);

                double len1 = norm(Point2f(l1[2], l1[3]) - Point2f(l1[0], l1[1]));
                double len2 = norm(Point2f(l2[2], l2[3]) - Point2f(l2[0], l2[1]));
                double avgLength = (len1 + len2) / 2.0;

                double diffVal = diff;
                double angleLine = line1.theta - CV_PI / 2;
                angleLine = angleLine * 180 / CV_PI;
                if (angleLine < 0)
                    angleLine += 180;

                parkingSpaces.emplace_back(center, Size2f(avgLength, diffVal), angleLine);
                i++;
            }
        }
    }
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