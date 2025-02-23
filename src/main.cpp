#include "ParkingSpaceClassifier.h"
#include "ParkingSpaceDetector.h"
#include "GroundTruthReader.h"
#include "Visualizer.h"
#include "tinyxml2.h"

#include <fstream>
#include <string>
#include <list>

#include <algorithm>
#include <random>

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
    
    for(int i=0; i<3; i++){
        trueOccupancyStatus.pop_back();
        trueParkingSpaces.pop_back();
    }
    

    // 2. Draw the parking spaces based on the occupancy detected using the isOccupied function
    std::vector<bool> occupancyStatus;
    std::vector<cv::RotatedRect> parkingSpaces = extractBoundingBoxesFromXML(xmlFilePath, occupancyStatus);

    for(int i=0; i<3; i++){
        parkingSpaces.pop_back();
    }
    
    // Shuffle the vector
    // std::random_device rd;
    // std::mt19937 g(rd());
    // std::shuffle(parkingSpaces.begin(), parkingSpaces.end(), g);
    
    ParkingSpaceClassifier classifier(0.3); // Initialize the classifier with an empty threshold of 0.4
    classifier.classifyParkingSpaces(parkingLotImage,parkingLotEmpty, parkingSpaces, occupancyStatus); 
    
    classifier.calculateMetrics(trueOccupancyStatus, occupancyStatus);
    
    Visualizer visualizer(450, 350, parkingSpaces, occupancyStatus);
    visualizer.drawParkingSpaces(imageFromDetection, occupancyStatus);

    // Determine the maximum width and height that can fit on the screen
    int screenHeight = 400;  // Example screen height
    int maxImageHeight = std::min(imageFromXML.rows, screenHeight);
    
    // Resize images to fit within the screen height
    double scaleFactor = static_cast<double>(maxImageHeight) / imageFromXML.rows;

    cv::resize(imageFromDetection, imageFromDetection, cv::Size(), scaleFactor, scaleFactor);

    
    if (occupancyStatus.size() < 37) {
        cerr << "Error: Occupancy vector size must be 38!" << endl;
        return -1;
    }
    // Create the minimap with occupancy data
    Mat empty_minimap = visualizer.createMockMinimap();
    Mat minimap = visualizer.updateMinimap(occupancyStatus);
    cv::Mat outputImage = visualizer.overlaySmallOnLarge(imageFromDetection, minimap);

    // Generate the output filename using stringstream
    std::string outputDir = "out";
    std::ostringstream filename;
    filename  << outputDir << "/outputS" << sequence << "F" << img_num << ".jpg";

    // Save the image with the generated filename
    bool success = cv::imwrite(filename.str(), outputImage);

    // Check if saving was successful
    if (!success) {
        std::cerr << "Failed to save the image" << std::endl;
        return 1;
    }

    std::cout << "Image saved successfully as " << filename.str() << std::endl;

    return 0;
}
