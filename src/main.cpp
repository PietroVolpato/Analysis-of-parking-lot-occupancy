#include <opencv2/opencv.hpp>
#include "tinyxml2.h"
#include <string>

using namespace cv;
using namespace tinyxml2;

// Function to draw the true parking spaces from an XML file
void drawTrueParkingSpaces(Mat &image, const std::string &xmlFilePath) {
    XMLDocument xmlDoc;
    XMLError eResult = xmlDoc.LoadFile(xmlFilePath.c_str());

    if (eResult != XML_SUCCESS) {
        std::cerr << "Error: Unable to load XML file!" << std::endl;
        return;
    }

    XMLElement* root = xmlDoc.FirstChildElement("parking");
    if (root == nullptr) {
        std::cerr << "Error: Invalid XML format!" << std::endl;
        return;
    }

    for (XMLElement* spaceElement = root->FirstChildElement("space"); spaceElement != nullptr; spaceElement = spaceElement->NextSiblingElement("space")) {
        int occupied;
        spaceElement->QueryIntAttribute("occupied", &occupied);

        XMLElement* rotatedRectElement = spaceElement->FirstChildElement("rotatedRect");
        if (rotatedRectElement == nullptr) continue;

        float centerX, centerY, width, height, angle;
        rotatedRectElement->FirstChildElement("center")->QueryFloatAttribute("x", &centerX);
        rotatedRectElement->FirstChildElement("center")->QueryFloatAttribute("y", &centerY);
        rotatedRectElement->FirstChildElement("size")->QueryFloatAttribute("w", &width);
        rotatedRectElement->FirstChildElement("size")->QueryFloatAttribute("h", &height);
        rotatedRectElement->FirstChildElement("angle")->QueryFloatAttribute("d", &angle);

        // Create RotatedRect from the parsed data
        RotatedRect parkingSpace(Point2f(centerX, centerY), Size2f(width, height), angle);

        // Draw the bounding box
        Point2f vertices[4];
        parkingSpace.points(vertices);

        Scalar color = occupied ? Scalar(0, 0, 255) : Scalar(0, 255, 0); // Red for occupied, green for empty

        for (int i = 0; i < 4; ++i)
            line(image, vertices[i], vertices[(i + 1) % 4], color, 2);
    }
}

int main() {
    // Load the parking lot image
    Mat parkingLotImage = imread("data/sequence0/frames/2013-02-24_10_05_04.jpg");
    if (parkingLotImage.empty()) {
        std::cerr << "Error: Unable to load image!" << std::endl;
        return -1;
    }

    // Draw the true parking spaces on the image using the XML file
    std::string xmlFilePath = "data/sequence0/bounding_boxes/2013-02-24_10_05_04.xml";
    drawTrueParkingSpaces(parkingLotImage, xmlFilePath);

    // Display the result
    imshow("True Parking Space Classification", parkingLotImage);
    waitKey(0);

    return 0;
}
