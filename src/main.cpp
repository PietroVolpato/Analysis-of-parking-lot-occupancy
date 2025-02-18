#include "ParkingSpaceClassifier.h"
#include "ParkingSpaceDetector.h"
#include "GroundTruthReader.h"
#include "Visualizer.h"
#include "tinyxml2.h"
#include <string>
#include <list>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {

    if (argc < 2) {
        cout << "Please provide sequence and image number." << endl;
        return 1;
    }

    ParkingSpaceDetector detector;
    // Loading the seq0 image for comparison
    vector<Mat> imgVectorSeq0 = detector.loadImages(0);
    // Load the images
    int sequence = stoi(argv[1]);
    vector<Mat> imgVector = detector.loadImages(sequence);

    // Contrast stretch the images
    int img_num = stoi(argv[2]);
    Mat parkingLotEmpty = imgVectorSeq0[img_num];
    Mat parkingLotImage = imgVector[img_num];
    
    // Path to the XML file
    std::vector<String> xmlFilePaths = loadXmlAddress(sequence);
    std::string xmlFilePath = xmlFilePaths[img_num];


    // Clone the image to create a separate copy for each method of occupancy detection
    cv::Mat imageFromXML = parkingLotImage.clone();
    cv::Mat imageFromDetection = parkingLotImage.clone();

    // 1. Draw the parking spaces based on the XML file (using the occupancy status from the XML file)
    std::vector<bool> trueOccupancyStatus;
    std::vector<cv::RotatedRect> trueParkingSpaces = extractBoundingBoxesFromXML(xmlFilePath, trueOccupancyStatus);
    
    Visualizer visualizer;
    visualizer.drawParkingSpaces(imageFromXML, trueParkingSpaces, trueOccupancyStatus);

    // 2. Draw the parking spaces based on the occupancy detected using the isOccupied function
    std::vector<bool> occupancyStatus;
    std::vector<cv::RotatedRect> parkingSpaces = extractBoundingBoxesFromXML(xmlFilePath, occupancyStatus);
    
    ParkingSpaceClassifier classifier(0.4); // Initialize the classifier with an empty threshold of 0.4
    classifier.classifyParkingSpaces(parkingLotImage,parkingLotEmpty, parkingSpaces, occupancyStatus);  
    visualizer.drawParkingSpaces(imageFromDetection, parkingSpaces, occupancyStatus);

    // Determine the maximum width and height that can fit on the screen
    int screenHeight = 400;  // Example screen height
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

    
    if (occupancyStatus.size() < 38) {
        cerr << "Error: Occupancy vector size must be 38!" << endl;
        cout << occupancyStatus.size() << endl;
        return -1;
    }
    // Create the minimap with occupancy data
    Mat minimap = visualizer.createMockMinimap(occupancyStatus);
    cv::imshow("minimap", minimap);
    cv::waitKey(0);

    return 0;
}
