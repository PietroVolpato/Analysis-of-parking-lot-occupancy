#include "ParkingSpaceDetector.h"
#include "CarSegmenter.h"
#include <fstream>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    ParkingSpaceDetector detector;

    // Load the images
    int sequence = 0;
    vector<Mat> emptyImgVector = detector.loadImages(sequence);

    // Preprocess the images
    vector<Mat> preprocessedImgVector;
    for (const auto& img : emptyImgVector) {
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
    for (const auto& img : preprocessedImgVector) {
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
    for (size_t i = 0; i < emptyImgVector.size(); ++i) {
        Mat img = emptyImgVector[i];
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

    // Draw the clustered lines
    for (size_t i = 0; i < emptyImgVector.size(); ++i) {
        Mat img = emptyImgVector[i];
        const auto& clusters = clustersVector[i];
        auto& cluster = clusters.first;
        detector.drawLines(img, cluster);
        imgWithLinesVector.push_back(img);
    }

    // Detect the parking spaces
    vector<vector<RotatedRect>> parkingSpacesVector;
    for (const auto clusters : clustersVector) {
        parkingSpacesVector.push_back(detector.detectParkingSpaces(clusters));
    }

    // Draw the parking spaces
    vector<Mat> imgWithParkingSpacesVector;
    for (size_t i = 0; i < emptyImgVector.size(); ++i) {
        Mat img = emptyImgVector[i];
        const auto& parkingSpaces = parkingSpacesVector[i];
        Mat imgWithParkingSpaces = detector.drawParkingSpaces(img, parkingSpaces);
        imgWithParkingSpacesVector.push_back(imgWithParkingSpaces);
    }

    // Draw the first cluster of the first image
    // Mat img = emptyImgVector[0];
    // const auto& cluster = clustersVector[0][0];
    // detector.drawLines(img, cluster);
    // imshow("Cluster 1", img);
    // waitKey(0);

    // Show the images
    for (const auto& img: imgWithLinesVector) {
    // for (const auto& img: imgWithParkingSpacesVector) {
        detector.showImage(img);
    }

    // CarSegmenter segmenter;
    
    // // Load the images
    // sequence = 2;
    // vector<Mat> imgVector = segmenter.loadImages(sequence);

    // // Preprocess the images
    // vector<Mat> preprocessedImgVector;
    // for (const auto& img : imgVector) {
    //     preprocessedImgVector.push_back(segmenter.preprocessImage(img, "gamma"));
    // }

    // // Create the average image
    // Mat avgImg = segmenter.createAvgImg(emptyImgVector);

    // // Convert the average image to grayscale
    // Mat grayAvgImg = segmenter.preprocessImage(avgImg, "gray");

    // // Compute the difference between the average image and the images
    // vector<Mat> diffImgVector;
    // for (const auto& img : preprocessedImgVector) {
    //     diffImgVector.push_back(segmenter.differenceImage(grayAvgImg, img));
    // }

    // // Analyze the difference images
    // for (auto& img : diffImgVector) {
    //     // img = segmenter.analyzeImage(img);
    // }

    // // // Find the contours in the difference images
    // // vector<pair<vector<vector<Point>>, vector<Vec4i>>> contoursVector;
    // // for (const auto& img : diffImgVector) {
    // //     contoursVector.push_back(segmenter.findContoursImg(img));
    // // }

    // // // Draw the contours
    // // for (size_t i = 0; i < diffImgVector.size(); ++i) {
    // //     Mat img = imgVector[i];
    // //     const auto& contours = contoursVector[i].first;
    // //     const auto& hierarchy = contoursVector[i].second;
    // //     segmenter.drawContoursImg(img, contours, hierarchy);
    // //     segmenter.showImages(img);
    // // }

    // // Show the images
    // for (const auto& img: diffImgVector) {
    //     segmenter.showImages(img);
    // }
    
    

    return 0;
}