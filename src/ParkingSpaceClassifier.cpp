#include "ParkingSpaceClassifier.h"
#include "tinyxml2.h"
#include <iostream>

using namespace cv;
using namespace tinyxml2;

// Function to create a bounding box
RotatedRect createBoundingBox(const Point2f& center, const Size2f& size, float angle) {
    return RotatedRect(center, size, angle);
}

void Thresholding(Mat &img) {
    Mat grayImg;
    cvtColor(img, grayImg, COLOR_BGR2GRAY);
    int lower_threshold = 70;
    int upper_threshold = 95;

    Mat lower_mask;
    threshold(grayImg, lower_mask, lower_threshold, 255, THRESH_BINARY);
    Mat upper_mask;
    threshold(grayImg, upper_mask, upper_threshold, 255, THRESH_BINARY_INV);
    Mat binaryImg;
    binaryImg = lower_mask & upper_mask;
    imshow("lower Threshold", lower_mask);
    imshow("Upper Threshold", upper_mask);
    //imshow("Filtered Image", binaryImg);
    waitKey(0);
}
// Function to determine if a parking space is occupied
bool isOccupied(const Mat &roi) {
    Mat grayRoi;
    cvtColor(roi, grayRoi, COLOR_BGR2GRAY);
    //imshow("Bbox", roi);
    //waitKey(0);
    // Define the lower and upper thresholds
    int lower_threshold = 70;
    int upper_threshold = 95;
    
    Mat lower_mask;
    threshold(grayRoi, lower_mask, lower_threshold, 255, THRESH_BINARY);
    Mat upper_mask;
    threshold(grayRoi, upper_mask, upper_threshold, 255, THRESH_BINARY_INV);
    Mat binaryRoi;
    binaryRoi = lower_mask & upper_mask;
    // threshold(grayRoi, binaryRoi, 0, 255, THRESH_BINARY | THRESH_OTSU);
    
    int nonZeroCount = countNonZero(binaryRoi);

    return nonZeroCount > 0.2 * binaryRoi.total(); // Adjust the threshold as necessary
}

// Function to classify parking spaces
void classifyParkingSpaces(const Mat &parkingLotImage, std::vector<RotatedRect> &parkingSpaces, std::vector<bool> &occupancyStatus) {
    for (size_t i = 0; i < parkingSpaces.size(); ++i) {
        Mat roi;
        getRectSubPix(parkingLotImage, parkingSpaces[i].size, parkingSpaces[i].center, roi);

        occupancyStatus[i] = isOccupied(roi);
    }
}

// Function to draw the parking spaces on the image
void drawParkingSpaces(Mat &image, const std::vector<RotatedRect> &parkingSpaces, const std::vector<bool> &occupancyStatus) {
    for (size_t i = 0; i < parkingSpaces.size(); ++i) {
        Point2f vertices[4];
        parkingSpaces[i].points(vertices);
        
        Scalar color = occupancyStatus[i] ? Scalar(0, 0, 255) : Scalar(0, 255, 0);

        for (int j = 0; j < 4; j++)
            line(image, vertices[j], vertices[(j+1)%4], color, 2);
    }
}

// Function to extract bounding boxes and their occupancy status from an XML file
std::vector<RotatedRect> extractBoundingBoxesFromXML(const std::string &xmlFilePath, std::vector<bool> &occupancyStatus) {
    std::vector<RotatedRect> boundingBoxes;

    XMLDocument xmlDoc;
    XMLError eResult = xmlDoc.LoadFile(xmlFilePath.c_str());

    if (eResult != XML_SUCCESS) {
        std::cerr << "Error: Unable to load XML file!" << std::endl;
        return boundingBoxes;
    }

    XMLElement* root = xmlDoc.FirstChildElement("parking");
    if (root == nullptr) {
        std::cerr << "Error: Invalid XML format!" << std::endl;
        return boundingBoxes;
    }

    for (XMLElement* spaceElement = root->FirstChildElement("space"); spaceElement != nullptr; spaceElement = spaceElement->NextSiblingElement("space")) {
        int occupied;
        spaceElement->QueryIntAttribute("occupied", &occupied);
        occupancyStatus.push_back(occupied == 1);

        XMLElement* rotatedRectElement = spaceElement->FirstChildElement("rotatedRect");
        if (rotatedRectElement == nullptr) continue;

        float centerX, centerY, width, height, angle;
        rotatedRectElement->FirstChildElement("center")->QueryFloatAttribute("x", &centerX);
        rotatedRectElement->FirstChildElement("center")->QueryFloatAttribute("y", &centerY);
        rotatedRectElement->FirstChildElement("size")->QueryFloatAttribute("w", &width);
        rotatedRectElement->FirstChildElement("size")->QueryFloatAttribute("h", &height);
        rotatedRectElement->FirstChildElement("angle")->QueryFloatAttribute("d", &angle);

        boundingBoxes.push_back(RotatedRect(Point2f(centerX, centerY), Size2f(width, height), angle));
    }

    return boundingBoxes;
}

// Function to draw the true parking spaces from an XML file
void drawTrueParkingSpaces(Mat &image, const std::string &xmlFilePath) {
    std::vector<bool> occupancyStatus;
    std::vector<RotatedRect> parkingSpaces = extractBoundingBoxesFromXML(xmlFilePath, occupancyStatus);
    drawParkingSpaces(image, parkingSpaces, occupancyStatus);
}
