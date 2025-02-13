#include "ParkingSpaceClassifier.h"
#include "ParkingSpaceDetector.h"
#include "GroundTruthReader.h"
#include "Visualizer.h"
#include "tinyxml2.h"
#include <string>
#include <list>

using namespace cv;


int main() {
    // Loading the seq0 image for comparison
    std::vector<Mat> imgVectorSeq0 = loadImages(0);
    // Load the images
    int sequence = 2;
    int img_num = 4;
    std::vector<Mat> imgVector = loadImages(sequence);
    
    Mat parkingLotEmpty = imgVectorSeq0[img_num];
    Mat parkingLotImage = imgVector[img_num];
    
    // Path to the XML file
    std::vector<String> xmlFilePaths = loadXmlAddress(sequence);
    std::string xmlFilePath = xmlFilePaths[img_num];


    // Clone the image to create a separate copy for each method of occupancy detection
    cv::Mat imageFromXML = parkingLotImage.clone();

    // 1. Draw the parking spaces based on the XML file (using the occupancy status from the XML file)
    std::vector<bool> trueOccupancyStatus;
    std::vector<cv::RotatedRect> trueParkingSpaces = extractBoundingBoxesFromXML(xmlFilePath, trueOccupancyStatus);
    drawParkingSpaces(imageFromXML, trueParkingSpaces, trueOccupancyStatus);


    // Determine the maximum width and height that can fit on the screen
    int screenHeight = 400;  // Example screen height
    int maxImageHeight = std::min(imageFromXML.rows, screenHeight);
    
    // Resize images to fit within the screen height
    double scaleFactor = static_cast<double>(maxImageHeight) / imageFromXML.rows;

    cv::resize(imageFromXML, imageFromXML, cv::Size(), scaleFactor, scaleFactor);

    // Display the combined result
    cv::imshow("Parking Space Occupancy Comparison", imageFromXML);
    // Create a minimap (e.g., 500x500 pixels)
    // Create a mock minimap that’s 500x300 pixels
    cv::Mat minimap = createMockMinimap(500, 300);
    // Display the minimap.
    cv::imshow("minimap", minimap);
    cv::waitKey(0);

    return 0;
}
