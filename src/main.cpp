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

    // // Preprocess the images
    // vector<Mat> preprocessedImgVector;
    // for (const auto& img : emptyImgVector) {
    //     preprocessedImgVector.push_back(detector.preprocessImage(img));
    // }

    // // Detect the edges in the images
    // vector<Mat> edgesImgVector;
    // for (const auto& img : preprocessedImgVector) {
    //     // edgesImgVector.push_back(detector.detectEdges(img, 800, 2400, 5));
    //     edgesImgVector.push_back(detector.detectEdges(img, 80, 240, 3));
    // }

    // // Detect the lines in the images
    // vector<vector<Vec4i>> linesVector;
    // for (const auto& img : preprocessedImgVector) {
    //     // linesVector.push_back(detector.detectLines(img, 10, 15, 10));
    //     linesVector.push_back(detector.detectLines(img, 25, 25, 15));
    // }

    // // Compute the line parameters
    // vector<vector<LineParams>> lineParamsVector;
    // for (const auto& lines : linesVector) {
    //     lineParamsVector.push_back(detector.computeLineParams(lines));
    // }

    // // Filter the lines
    // vector<vector<LineParams>> filteredLinesVector;
    // for (auto& lineParams : lineParamsVector) {
    //     filteredLinesVector.push_back(detector.filterLines(lineParams));
    // }

    // // Draw the detected lines
    // vector<Mat> imgWithLinesVector;
    // for (size_t i = 0; i < emptyImgVector.size(); ++i) {
    //     Mat img = emptyImgVector[i];
    //     // const auto& lines = lineParamsVector[i];
    //     const auto& lines = filteredLinesVector[i];
    //     // detector.drawLines(img, lines);
    //     imgWithLinesVector.push_back(img);
    // }

    // // Cluster the lines
    // vector<pair<vector<LineParams>, vector<LineParams>>> clustersVector;
    // for (const auto& lines : filteredLinesVector) {
    //     clustersVector.push_back(detector.clusterLines(lines));
    // }

    // // Draw the clustered lines
    // for (size_t i = 0; i < emptyImgVector.size(); ++i) {
    //     Mat img = emptyImgVector[i];
    //     const auto& clusters = clustersVector[i];
    //     auto& cluster = clusters.first;
    //     detector.drawLines(img, cluster);
    //     imgWithLinesVector.push_back(img);
    // }

    // // Detect the parking spaces
    // vector<vector<RotatedRect>> parkingSpacesVector;
    // for (const auto clusters : clustersVector) {
    //     parkingSpacesVector.push_back(detector.detectParkingSpaces(clusters));
    // }

    // // Draw the parking spaces
    // vector<Mat> imgWithParkingSpacesVector;
    // for (size_t i = 0; i < emptyImgVector.size(); ++i) {
    //     Mat img = emptyImgVector[i];
    //     const auto& parkingSpaces = parkingSpacesVector[i];
    //     Mat imgWithParkingSpaces = detector.drawParkingSpaces(img, parkingSpaces);
    //     imgWithParkingSpacesVector.push_back(imgWithParkingSpaces);
    // }

    // // Draw the first cluster of the first image
    // // Mat img = emptyImgVector[0];
    // // const auto& cluster = clustersVector[0][0];
    // // detector.drawLines(img, cluster);
    // // imshow("Cluster 1", img);
    // // waitKey(0);

    // // Show the images
    // for (const auto& img: imgWithLinesVector) {
    // // for (const auto& img: imgWithParkingSpacesVector) {
    //     detector.showImage(img);
    // }

    CarSegmenter segmenter;
    
    // Load the images
    sequence = 1;
    vector<Mat> imgVector = segmenter.loadImages(sequence);

    // Preprocess the images
    vector<Mat> preprocessedImgVector;
    for (const auto& img : imgVector) {
        preprocessedImgVector.push_back(segmenter.preprocessImage(img, "gray"));
    }

    // String filePath = "../data/sequence0/bounding_boxes/2013-02-24_10_05_04.xml";
    // vector<RotatedRect> bboxes = segmenter.getBBoxes(filePath);
    // segmenter.drawBBoxes(imgVector[1], bboxes);
    // segmenter.showImages(imgVector[1]);

    // Use the background subtraction method
    vector<Mat> trainingVector;
    for (const auto& img: emptyImgVector) {
        trainingVector.push_back(segmenter.preprocessImage(img, "gray"));
        trainingVector.push_back(segmenter.preprocessImage(img, "gamma"));
        trainingVector.push_back(segmenter.preprocessImage(img, "equalize"));
    }

    segmenter.trainBg(trainingVector);

    vector<Mat> maskVector;
    for (const auto& img : preprocessedImgVector) {
        maskVector.push_back(segmenter.applyBg(img));
    }

    vector<Mat> enhancedMaskVector;
    for (const auto& mask : maskVector) {
        enhancedMaskVector.push_back(segmenter.enhanceMask(mask));
    }

    // Find the contours in the images
    vector<vector<vector<Point>>> contoursVector;
    vector<vector<Vec4i>> hierarchyVector;
    for (const auto& mask : enhancedMaskVector) {
        auto [contours, hierarchy] = segmenter.findContoursImg(mask);
        contoursVector.push_back(contours);
        hierarchyVector.push_back(hierarchy);
    }

    // // Draw the contours
    // for (size_t i = 0; i < imgVector.size(); ++i) {
    //     Mat img = imgVector[i];
    //     const auto& contours = contoursVector[i];
    //     const auto& hierarchy = hierarchyVector[i];
    //     segmenter.drawContoursImg(img, contours, hierarchy);
    //     segmenter.showImages(img);
    // }

    // Find the bounding boxes
    vector<vector<RotatedRect>> bboxesVector;
    for (size_t i = 0; i < imgVector.size(); ++i) {
        const auto& contours = contoursVector[i];
        bboxesVector.push_back(segmenter.findBBoxes(contours));
    }

    // Filter the bounding boxes
    for (auto& bboxes : bboxesVector) {
        bboxes = segmenter.filterBBoxes(bboxes);
    }

    // Draw the bounding boxes
    // for (size_t i = 0; i < imgVector.size(); ++i) {
    //     Mat img = imgVector[i];
    //     const auto& bboxes = bboxesVector[i];
    //     segmenter.drawBBoxes(img, bboxes);
    //     segmenter.showImages(img);
    // }

    // Segment the cars
    vector<Mat> segmentedCarsVector;
    for (size_t i = 0; i < imgVector.size(); ++i) {
        const auto& bboxes = bboxesVector[i];
        const auto& mask = enhancedMaskVector[i];
        const auto& img = imgVector[i];
        segmentedCarsVector.push_back(segmenter.segmentCar(bboxes, mask, img));
    }

    // Show the images
    for (const auto& img: segmentedCarsVector) {
        segmenter.showImages(img);
    }
    
    

    return 0;
}