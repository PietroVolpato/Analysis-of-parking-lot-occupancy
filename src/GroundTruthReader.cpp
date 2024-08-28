#include "GroundTruthReader.h"

using namespace cv;
using namespace tinyxml2;

// Function to load the address
std::vector<String> loadXmlAddress(int sequence) {
    std::vector<String> fileNames;

    if (sequence >= 0 && sequence <= 5) {
        String path = "./data/sequence" + std::to_string(sequence) + "/bounding_boxes";
        glob(path, fileNames);
    } else {
        // Handle the error case if needed
        std::cerr << "Invalid sequence number: " << sequence << std::endl;
    }

    return fileNames;
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
