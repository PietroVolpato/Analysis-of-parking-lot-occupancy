#include "ParkingSpaceClassifier.h"
#include "GroundTruthReader.h"
#include <opencv2/opencv.hpp>
#include "tinyxml2.h"
#include <string>


int main() {
    // Load the parking lot image
    cv::Mat parkingLotImage = cv::imread("data/sequence1/frames/2013-02-22_07_10_01.png");
    if (parkingLotImage.empty()) {
        std::cerr << "Error: Unable to load image!" << std::endl;
        return -1;
    }
    
    // Clone the image to create a separate copy for each method of occupancy detection
    cv::Mat imageFromXML = parkingLotImage.clone();
    cv::Mat imageFromDetection = parkingLotImage.clone();

    // Path to the XML file
    std::string xmlFilePath = "data/sequence1/bounding_boxes/2013-02-22_07_10_01.xml";

    // 1. Draw the parking spaces based on the XML file (using the occupancy status from the XML file)
    std::vector<bool> trueOccupancyStatus;
    std::vector<cv::RotatedRect> trueParkingSpaces = extractBoundingBoxesFromXML(xmlFilePath, trueOccupancyStatus);
    drawParkingSpaces(imageFromXML, trueParkingSpaces, trueOccupancyStatus);

    // 2. Draw the parking spaces based on the occupancy detected using the isOccupied function
    std::vector<bool> occupancyStatus;
    std::vector<cv::RotatedRect> parkingSpaces = extractBoundingBoxesFromXML(xmlFilePath, occupancyStatus);
    
    classifyParkingSpaces(parkingLotImage, parkingSpaces, occupancyStatus);  // Re-classify using K-Means
    drawParkingSpaces(imageFromDetection, parkingSpaces, occupancyStatus);

    // Determine the maximum width and height that can fit on the screen
    int screenHeight = 400;  // Example screen height, adjust as needed
    int maxImageHeight = std::min(imageFromXML.rows, screenHeight);
    
    // Resize images to fit within the screen height
    double scaleFactor = static_cast<double>(maxImageHeight) / imageFromXML.rows;

    cv::resize(imageFromXML, imageFromXML, cv::Size(), scaleFactor, scaleFactor);
    cv::resize(imageFromDetection, imageFromDetection, cv::Size(), scaleFactor, scaleFactor);

    // Combine the two images side by side for comparison
    cv::Mat combined;
    cv::hconcat(imageFromXML, imageFromDetection, combined);

    // Display the combined result
    cv::imshow("Parking Space Occupancy Comparison", combined);
    cv::waitKey(0);

    return 0;
}
