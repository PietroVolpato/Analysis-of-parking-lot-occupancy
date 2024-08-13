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
    for (int i = 0; i < imgs.size(); i++) {
        Mat grayImg;
        cvtColor(imgs[i], grayImg, COLOR_BGR2GRAY);
        
        Mat blurImg;
        bilateralFilter(grayImg, blurImg, 9, 150, 150);

        Mat processedImg;
        equalizeHist(blurImg, processedImg);

        processedImages.push_back(processedImg);
    }

    return processedImages;
}

void showImages (const std::vector<Mat>& imgs) {
    for (int i = 0; i < imgs.size(); i++) {
        imshow("Image", imgs[i]);
        waitKey(0);
    }
}

std::vector<Mat> detectEdges (const std::vector<Mat>& imgs) {
    std::vector<Mat> edges;
    Mat img;
    for (int i = 0; i < imgs.size(); i++) {
        Canny(imgs[i], img, 100, 200);
        edges.push_back(img);
    }

    return edges;
}

std::vector<std::vector<Vec4i>> detectLines (const std::vector<Mat>& imgs) {
    std::vector<std::vector<Vec4i>> lines;
    std::vector<Vec4i> line;
    for (int i = 0; i < imgs.size(); i++) {
        HoughLinesP(imgs[i], line, 1, CV_PI/180, 50, 50, 10);
        lines.push_back(line);
    }

    return lines;
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