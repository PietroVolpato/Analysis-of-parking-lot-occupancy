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
        Mat img = imread(fileNames[i]);
        if (img.empty()) {
            std::cerr << "Error loading image: " << fileNames[i] << std::endl;
            continue;
        }
        imgs.push_back(img);
    }

    return imgs;
}

std::vector<Mat> preprocessImages (const std::vector<Mat>& imgs) {
    std::vector<Mat> processedImages;
    processedImages.reserve(imgs.size());
    for (const auto& img: imgs) {
        Mat grayImg;
        cvtColor(img, grayImg, COLOR_BGR2GRAY);
        
        Mat blurImg;
        GaussianBlur(grayImg, blurImg, Size(5, 5), 0);

        Mat processedImg;
        equalizeHist(blurImg, processedImg);

        processedImages.push_back(processedImg);
    }

    return processedImages;
}

void showImages (const std::vector<Mat>& imgs) {
    for (const auto& img: imgs) {
        imshow("Image", img);
        waitKey(0);
    }
}

std::vector<Mat> detectEdges (const std::vector<Mat>& imgs, double lowThreshold, double highThreshold) {
    std::vector<Mat> edges;
    edges.reserve(imgs.size());
    for (const auto& img: imgs) {
        Mat edge;
        Canny(img, edge, lowThreshold, highThreshold);
        edges.push_back(edge);
    }

    return edges;
}

std::vector<std::vector<Vec4i>> detectLines (const std::vector<Mat>& imgs, int threshold, double minLineLength, double maxLineGap) {
    std::vector<std::vector<Vec4i>> lines;
    std::vector<Vec4i> line;
    for (const auto& img: imgs) {
        HoughLinesP(img, line, 1, CV_PI/180, threshold, minLineLength, maxLineGap);
        lines.push_back(line);
    }

    return lines;
}

// Filter the lines to only include horizontal and vertical lines
std::vector<std::vector<Vec4i>> filterLines (const std::vector<std::vector<Vec4i>>& linesVector) {
    std::vector<std::vector<Vec4i>> filteredLinesVector;
    const double angleThreshold = 5.0;

    for (const auto& lines: linesVector) {
        std::vector<Vec4i> filteredLines;
        for (const auto& line: lines) {
            double dx1 = line[2] - line[0];
            double dy1 = line[3] - line[1];
            double length1 = sqrt(dx1*dx1 + dy1*dy1);
            double angle1 = atan2(dy1, dx1) * 180 / CV_PI;

            // Filter for length
            if (length1 < 100) {
                // Filter for angle
                if ((angle1 > 30 && angle1 < 60) || (angle1 > 120 && angle1 < 150)) {
                    bool isParallel = false;

                    // Check if the line is parallel to any other line
                    for (const auto& l: filteredLines) {
                        double dx2 = l[2] - l[0];
                        double dy2 = l[3] - l[1];
                        double length2 = sqrt(dx2*dx2 + dy2*dy2);
                        double angle2 = atan2(dy2, dx2) * 180 / CV_PI;

                        if (abs(angle1 - angle2) < angleThreshold) {
                            isParallel = true;
                            break;
                        }
                    }

                    if (isParallel || filteredLines.empty()) {
                        filteredLines.push_back(line);
                    }
                }
            }

        }

        filteredLinesVector.push_back(filteredLines);
    }

    return filteredLinesVector;
}



void drawLines (std::vector<Mat>& imgs, const std::vector<std::vector<Vec4i>>& lines) {
    for (const auto& img: imgs) {
        for (const auto& line: lines) {
            for (const auto& l: line) {
                cv::line(img, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);
            }
        }
    }
}

void drawBoundingBoxes (std::vector<Mat>& imgs, const std::vector<std::vector<Vec4i>>& linesVector) {
}