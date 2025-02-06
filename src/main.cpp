#include "ParkingSpaceDetector.h"
// #include "ParkingSpaceClassifier.h"
// #include "CarSegmenter.h"
// #include "Visualizer.h"

using namespace cv;

int main (int argc, char** argv) {
    // Load the images
    int sequence = 0;
    std::vector<Mat> imgVector = loadImages(sequence);

    // Preprocess the images
    std::vector<Mat> edgesImgVector;
    for (const auto& img : imgVector) {
       edgesImgVector.push_back(preprocessImage(img));
    }

    // Detect the lines in the images
    std::vector<std::vector<Vec4i>> linesVector;
    for (const auto& img : edgesImgVector) {
        linesVector.push_back(detectLines(img, 30, 30, 10));
    }

    // Compute the line parameters
    std::vector<std::vector<LineParams>> lineParamsVector;
    for (const auto& lines : linesVector) {
        lineParamsVector.push_back(computeLineParams(lines));
    }

    // for (size_t i = 0; i < edgesImgVector.size() && i < imgVector.size(); ++i) {
    //     findContours(edgesImgVector[i], imgVector[i]);
    // }

    // Filter the lines
    std::vector<std::vector<LineParams>> filteredLinesVector;
    for (auto& lineParams : lineParamsVector) {
        filteredLinesVector.push_back(filterLines(lineParams));
    }

    std::vector<Mat> imgWithLinesVector;
    for (size_t i = 0; i < imgVector.size(); ++i) {
        Mat img = imgVector[i];
        const auto& lines = lineParamsVector[i];
        drawLines(img, lines);
        imgWithLinesVector.push_back(img);
    }

    // Detect the parking spaces
    std::vector<std::vector<RotatedRect>> parkingSpacesVector;
    for (const auto& lines : filteredLinesVector) {
        parkingSpacesVector.push_back(detectParkingSpaces(lines));
    }

    // Draw the parking spaces
    std::vector<Mat> imgWithParkingSpacesVector;
    for (size_t i = 0; i < imgVector.size(); ++i) {
        Mat img = imgVector[i];
        const auto& parkingSpaces = parkingSpacesVector[i];
        imgWithParkingSpacesVector.push_back(drawParkingSpaces(img, parkingSpaces));
    }

   
    // Show the images
    for (const auto& img : edgesImgVector) {
        showImage(img);
    }

    // for (const auto& img: imgVector) {
    //     prova(img);
    // }

    return 0;
}