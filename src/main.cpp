#include "ParkingSpaceDetector.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    ParkingSpaceDetector detector;

    // Load the images
    int sequence = 0;
    vector<Mat> imgVector = detector.loadImages(sequence);

    // Preprocess the images
    vector<Mat> preprocessedImgVector;
    for (const auto& img : imgVector) {
        preprocessedImgVector.push_back(detector.preprocessImage(img));
    }

    // Detect the edges in the images
    vector<Mat> edgesImgVector;
    for (const auto& img : preprocessedImgVector) {
        edgesImgVector.push_back(detector.detectEdges(img));
    }

    // Detect the lines in the images
    vector<vector<Vec4i>> linesVector;
    for (const auto& img : preprocessedImgVector) {
        linesVector.push_back(detector.detectLines(img, 50, 30, 10));
    }

    // Compute the line parameters
    vector<vector<LineParams>> lineParamsVector;
    for (const auto& lines : linesVector) {
        lineParamsVector.push_back(detector.computeLineParams(lines));
    }

    // Draw the detected lines
    vector<Mat> imgWithLinesVector;
    for (size_t i = 0; i < imgVector.size(); ++i) {
        Mat img = imgVector[i];
        const auto& lines = lineParamsVector[i];
        detector.drawLines(img, lines);
        imgWithLinesVector.push_back(img);
    }

    // Filter the lines
    vector<vector<LineParams>> filteredLinesVector;
    for (auto& lineParams : lineParamsVector) {
        filteredLinesVector.push_back(detector.filterLines(lineParams));
    }

    // Detect parking spaces based on the filtered lines (if desired)
    vector<RotatedRect> parkingSpaces = detector.detectParkingSpaces(filteredLinesVector[0]);

    // Show the images
    for (const auto& img : imgWithLinesVector) {
        detector.showImage(img);
    }

    return 0;
}