#include "ParkingSpaceDetector.h"
#include <fstream>

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
        // edgesImgVector.push_back(detector.detectEdges(img, 800, 2400, 5));
        edgesImgVector.push_back(detector.detectEdges(img, 80, 240, 3));
    }

    // Detect the lines in the images
    vector<vector<Vec4i>> linesVector;
    for (const auto& img : edgesImgVector) {
        // linesVector.push_back(detector.detectLines(img, 10, 15, 10));
        linesVector.push_back(detector.detectLines(img, 25, 25, 15));
    }

    // Compute the line parameters
    vector<vector<LineParams>> lineParamsVector;
    for (const auto& lines : linesVector) {
        lineParamsVector.push_back(detector.computeLineParams(lines));
    }

    // Filter the lines
    vector<vector<LineParams>> filteredLinesVector;
    for (auto& lineParams : lineParamsVector) {
        filteredLinesVector.push_back(detector.filterLines(lineParams));
    }

    // Draw the detected lines
    vector<Mat> imgWithLinesVector;
    for (size_t i = 0; i < imgVector.size(); ++i) {
        Mat img = imgVector[i];
        // const auto& lines = lineParamsVector[i];
        const auto& lines = filteredLinesVector[i];
        // detector.drawLines(img, lines);
        imgWithLinesVector.push_back(img);
    }

    // Cluster the lines
    vector<pair<vector<LineParams>, vector<LineParams>>> clustersVector;
    for (const auto& lines : filteredLinesVector) {
        clustersVector.push_back(detector.clusterLines(lines));
    }

    // // Draw the clustered lines
    // for (size_t i = 0; i < imgVector.size(); ++i) {
    //     Mat img = imgVector[i];
    //     const auto& clusters = clustersVector[i];
    //     for (const auto& cluster : clusters) {
    //         detector.drawLines(img, cluster);
    //     }
    //     imgWithLinesVector.push_back(img);
    // }

    // Detect the parking spaces
    vector<vector<RotatedRect>> parkingSpacesVector;
    for (const auto clusters : clustersVector) {
        parkingSpacesVector.push_back(detector.detectParkingSpaces(clusters));
    }

    // Draw the parking spaces
    vector<Mat> imgWithParkingSpacesVector;
    for (size_t i = 0; i < imgVector.size(); ++i) {
        Mat img = imgVector[i];
        const auto& parkingSpaces = parkingSpacesVector[i];
        Mat imgWithParkingSpaces = detector.drawParkingSpaces(img, parkingSpaces);
        imgWithParkingSpacesVector.push_back(imgWithParkingSpaces);
    }

    // Draw the first cluster of the first image
    // Mat img = imgVector[0];
    // const auto& cluster = clustersVector[0][0];
    // detector.drawLines(img, cluster);
    // imshow("Cluster 1", img);
    // waitKey(0);

    // Show the images
    // for (const auto& img: imgWithLinesVector) {
    for (const auto& img: imgWithParkingSpacesVector) {
        detector.showImage(img);
    }

    return 0;
}