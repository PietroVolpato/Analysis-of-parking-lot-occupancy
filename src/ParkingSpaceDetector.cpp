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

std::vector<Mat> preprocessImages (const std::vector<Mat>& imgs, int d, double sigmaColor, double sigmaSpace) {
    std::vector<Mat> processedImages;
    processedImages.reserve(imgs.size());
    for (const auto& img: imgs) {
        Mat grayImg;
        cvtColor(img, grayImg, COLOR_BGR2GRAY);
        
        Mat blurImg;
        bilateralFilter(grayImg, blurImg, d, sigmaColor, sigmaSpace);

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
    std::vector<std::vector<Vec4i>> filteredLines;
    const double tolerance = 10;
    for (const auto& lines: linesVector) {
        std::vector<Vec4i> filteredImgLines;
        for (const auto& line: lines) {
            double angle = std::atan2(line[3] - line[1], line[2] - line[0]) * 180 / CV_PI;
            if (std::abs(angle) < tolerance || std::abs(angle - 90) < tolerance) {
                filteredImgLines.push_back(line);
            }
            filteredLines.push_back(filteredImgLines);
        }
    }
    
    return filteredLines;
}

// Cluster the lines into groups of parallel lines
std::vector<std::vector<Vec4i>> clusterLines (const std::vector<std::vector<Vec4i>>& linesVector) {
    std::vector<std::vector<Vec4i>> clusteredLines;
    for (const auto& lines: linesVector) {
        std::vector<Vec4i> clusteredLine;
        if (lines.empty()) {
            clusteredLines.push_back(clusteredLine);
            continue;
        }

        Vec4i averageLine = lines[0];
        int count = 1;

        for (const auto& line: lines) {
            double angle1 = std::atan2(averageLine[3] - averageLine[1], averageLine[2] - averageLine[0]);
            double angle2 = std::atan2(line[3] - line[1], line[2] - line[0]);

            if (std::abs(angle1 - angle2) < CV_PI / 180 * 10) {
                averageLine[0] += line[0];
                averageLine[1] += line[1];
                averageLine[2] += line[2];
                averageLine[3] += line[3];
                count++;
            }
            else {
                averageLine[0] /= count;
                averageLine[1] /= count;
                averageLine[2] /= count;
                averageLine[3] /= count;
                clusteredLine.push_back(averageLine);
                averageLine = line;
                count = 1;
            }
        }

        averageLine[0] /= count;
        averageLine[1] /= count;
        averageLine[2] /= count;
        averageLine[3] /= count;
        clusteredLine.push_back(averageLine);

        clusteredLines.push_back(clusteredLine);
    }

    return clusteredLines;
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
    for (size_t i = 0; i < imgs.size(); i++) {
        for (size_t j = 0; j < linesVector[i].size(); j += 2) {
            if (j + 1 < linesVector[i].size()) {
                Vec4i line1 = linesVector[i][j];
                Vec4i line2 = linesVector[i][j + 1];

                // Midde point of the lines
                Point2f midpoint1 ((line1[0] + line1[2]) / 2, (line1[1] + line1[3]) / 2);
                Point2f midpoint2 ((line2[0] + line2[2]) / 2, (line2[1] + line2[3]) / 2);

                // Calculate the angle of the bbox
                double angle = std::atan2(midpoint2.y - midpoint1.y, midpoint2.x - midpoint1.x) * 180 / CV_PI;

                // Distance between the midpoints
                double width = cv::norm(midpoint1 - midpoint2);

                // Estimate the width and height of the bbox
                double height = 0.5 * width;

                // Center of the bbox
                Point2f center ((midpoint1.x + midpoint2.x) / 2, (midpoint1.y + midpoint2.y) / 2);

                // Create the bbox
                RotatedRect bbox (center, Size2f(width, height), angle);

                // Draw the bbox
                cv::Point2f vertices[4];
                bbox.points(vertices);  
                for (int k = 0; k < 4; k++)
                    cv::line(imgs[k], vertices[k], vertices[(k + 1) % 4], Scalar(0, 255, 0), 2);
            }
        }
    }
}