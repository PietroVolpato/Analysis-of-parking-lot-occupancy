#include "ParkingSpaceDetector.h"

using namespace cv;

std::vector<Mat> loadImages (int sequence) {
    std::vector<String> fileNames;
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

    std::vector<Mat> imgs;
    for (int i = 0; i < fileNames.size(); i++) {
        imgs.push_back(imread(fileNames[i]));
    }

    return imgs;
}

void showImages (const std::vector<Mat>& imgs) {
    for (int i = 0; i < imgs.size(); i++) {
        imshow("Image", imgs[i]);
        waitKey(0);
    }
}

Mat detectEdges (const Mat& img) {
    Mat edges;
    Canny(img, edges, 50, 150);
    return edges;
}

std::vector<Vec4i> detectLines (const Mat& img) {
    std::vector<Vec4i> lines;
    HoughLinesP(img, lines, 1, CV_PI/180, 50, 50, 10);
    
    return lines;
}

std::vector<std::vector<Point>> detectContours (const Mat& img) {
    std::vector<std::vector<Point>> contours;
    findContours(img, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    return contours;
}

double calculateAngle (const Vec4i& line) {
    double angle = atan2(line[3] - line[1], line[2] - line[0]) * 180 / CV_PI;
    return angle;
}

std::vector<std::vector<Vec4i>> filterLines (const std::vector<Vec4i>& lines) {
    const double angle_tollerance = 5.0;
    std::vector<std::vector<Vec4i>> parallelLines;
    for (size_t i = 0; i < lines.size(); i++) {
        std::vector<Vec4i> tempLines;
        double angle = calculateAngle(lines[i]);
        tempLines.push_back(lines[i]);
        for (size_t j = i+1; j < lines.size(); j++) {
            double angle2 = calculateAngle(lines[j]);
            if (std::abs(angle - angle2) < angle_tollerance) {
                tempLines.push_back(lines[j]);
            }
        }
    }

    return parallelLines;
}

void drawParkingSpaces (const Mat& img, const std::vector<Vec4i>& lines, const std::vector<std::vector<Point>>& contours) {
    // Draw the lines
    for (int i = 0; i < lines.size(); i++) {
        line(img, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]), Scalar(0, 0, 255), 3, LINE_AA);
    }

    // Draw the constours and the bouding boxes
    for (const auto& contour : contours) {
        RotatedRect box = minAreaRect(contour);
        Point2f vertices[4];
        box.points(vertices);
        for (int i = 0; i < 4; i++) {
            line(img, vertices[i], vertices[(i + 1) % 4], Scalar(255, 0, 0), 2, LINE_AA);
        }
    }
}

void drawBoundingBoxes (const Mat& img, const std::vector<std::vector<Vec4i>>& parallelLines, const std::vector<std::vector<Point>>& contours) {
    for (const auto& lines : parallelLines) {
       std::vector<Point> points;
        for (const auto& line : lines) {
              points.push_back(Point(line[0], line[1]));
              points.push_back(Point(line[2], line[3]));
        }

        // Create a bounding box for the lines
        Rect boundingBox = boundingRect(points);

        // Check if the bounding box contains any of the contours
        for (const auto& contour: contours) {
            Rect counterBoundingBox = boundingRect(contour);
            if ((counterBoundingBox & boundingBox).area() > 0) {
                RotatedRect box = minAreaRect(contour);
                Point2f vertices[4];
                box.points(vertices);
                for (int i = 0; i < 4; i++) {
                    line(img, vertices[i], vertices[(i + 1) % 4], Scalar(255, 0, 0), 2, LINE_AA);
                }
                
                break;
            }
        }
    }
}