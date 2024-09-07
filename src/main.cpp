#include "ParkingSpaceClassifier.h"
#include "ParkingSpaceDetector.h"
#include "GroundTruthReader.h"
#include "tinyxml2.h"
#include <string>

using namespace cv;


int main() {
    // Load the images
    int sequence = 1;
    int img_num = 2;
    std::vector<Mat> imgVector = loadImages(sequence);

     // Contrast stretch the images
    // std::vector<Mat> stretchedImgVector = constrastStretch(imgVector);
    
    cv::Mat parkingLotImage = imgVector[img_num];
    
    // Path to the XML file
    std::vector<String> xmlFilePaths = loadXmlAddress(sequence);
    std::string xmlFilePath = xmlFilePaths[img_num];


    // Clone the image to create a separate copy for each method of occupancy detection
    cv::Mat imageFromXML = parkingLotImage.clone();
    cv::Mat imageFromDetection = parkingLotImage.clone();

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
