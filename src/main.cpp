#include "ParkingSpaceClassifier.h"
#include "ParkingSpaceDetector.h"
#include "GroundTruthReader.h"
#include "CarSegmenter.h"
#include "Visualizer.h"
#include "Evaluator.h"
#include "Loader.h"
#include "tinyxml2.h"
#include <string>
#include <list>
#include <algorithm>
#include <random>

using namespace cv;
using namespace std;

struct ParkingSpace {
    int id;
    RotatedRect rect;
    bool occupied;
};

int main(int argc, char** argv) {
    // Create the loader
    Loader loader;

    // Load the empty images
    int sequence = 0;
    vector<Mat> emptyImgVector = loader.loadImagesFromSequence(0);

    // Load all the other images
    vector<int> sequences = {1, 2, 3, 4, 5};
    vector<Mat> imgVector;
    for (const int seq: sequences) {
        vector<Mat> imgs = loader.loadImagesFromSequence(seq);
        imgVector.insert(imgVector.end(), imgs.begin(), imgs.end());
    }

    // Load the ground truth bounding boxes
    vector<RotatedRect> groundtruthBBoxes = loader.getBBoxes("../data/sequence0/bounding_boxes/2013-02-24_10_05_04.xml");

    // Load all the masks for the evaluation
    vector<Mat> evaluationMaskVector;
    for (const int seq: sequences) {
        vector<Mat> masks = loader.loadMask(seq);
        evaluationMaskVector.insert(evaluationMaskVector.end(), masks.begin(), masks.end());
    }

    // Detect the parking spaces
    ParkingSpaceDetector detector;


    // Preprocess the images
    vector<Mat> preprocessedEmptyImgVector;
    for (const auto& img : emptyImgVector) {
        preprocessedEmptyImgVector.push_back(detector.preprocessImage(img));
    }

    // Detect the lines in the images
    vector<vector<Vec4i>> linesVector;
    for (const auto& img : preprocessedEmptyImgVector) {
        linesVector.push_back(detector.detectLines(img));
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
    vector<pair<vector<pair<LineParams, LineParams>>, vector<pair<LineParams, LineParams>>> > clustersVector;
    for (const auto& lines : filteredLinesVector) {
        clustersVector.push_back(detector.clusterLines(lines));
    }

    // // Draw the clustered lines
    // for (size_t i = 0; i < emptyImgVector.size(); ++i) {
    //     Mat img = emptyImgVector[i];
    //     const auto& clusters = clustersVector[i];
    //     auto& cluster = clusters.first;
    //     detector.drawLines(img, cluster);
    //     imgWithLinesVector.push_back(img);
    // }

    // Detect the parking spaces
    vector<vector<RotatedRect>> parkingSpacesVector;
    for (const auto clusters : clustersVector) {
        parkingSpacesVector.push_back(detector.detectParkingSpaces(clusters));
    }
    for (const auto& lines : filteredLinesVector) {
        // parkingSpacesVector.push_back(detector.detectParkingSpacesSimple(lines));
    }

    // Draw the parking spaces
    vector<Mat> imgWithParkingSpacesVector;
    for (size_t i = 0; i < emptyImgVector.size(); ++i) {
        Mat img = emptyImgVector[i];
        const auto& parkingSpaces = parkingSpacesVector[i];
        Mat imgWithParkingSpaces = detector.drawParkingSpaces(img, parkingSpaces);
        imgWithParkingSpacesVector.push_back(imgWithParkingSpaces);
    }

    // Show the images
    // for (const auto& img: imgWithLinesVector) {
    for (const auto& img: imgWithParkingSpacesVector) {
        detector.showImage(img);
    }

    // CarSegmenter segmenter;

    // // Preprocess the images
    // vector<Mat> preprocessedImgVector;
    // for (const auto& img : imgVector) {
    //     preprocessedImgVector.push_back(segmenter.preprocessImage(img, "gray"));
    // }

    // // Use the background subtraction method
    // vector<Mat> trainingVector;
    // for (const auto& img: emptyImgVector) {
    //     trainingVector.push_back(segmenter.preprocessImage(img, "gray"));
    //     trainingVector.push_back(segmenter.preprocessImage(img, "gamma"));
    //     trainingVector.push_back(segmenter.preprocessImage(img, "equalize"));
    // }

    // segmenter.trainBg(trainingVector);

    // vector<Mat> maskVector;
    // for (const auto& img : preprocessedImgVector) {
    //     maskVector.push_back(segmenter.applyBg(img));
    // }

    // vector<Mat> enhancedMaskVector;
    // for (const auto& mask : maskVector) {
    //     enhancedMaskVector.push_back(segmenter.enhanceMask(mask));
    // }

    // // Find the contours in the images
    // vector<vector<vector<Point>>> contoursVector;
    // for (const auto& mask : enhancedMaskVector) {
    //     auto contours = segmenter.findContoursSimple(mask);
    //     contoursVector.push_back(contours);
    // }

    // // // Draw the contours
    // // for (size_t i = 0; i < imgVector.size(); ++i) {
    // //     Mat img = imgVector[i];
    // //     const auto& contours = contoursVector[i];
    // //     // segmenter.drawContourSimple(img, contours);
    // //     // segmenter.showImages(img);
    // // }

    // // Find the bounding boxes
    // vector<vector<RotatedRect>> bboxesVector;
    // for (size_t i = 0; i < imgVector.size(); ++i) {
    //     const auto& contours = contoursVector[i];
    //     bboxesVector.push_back(segmenter.findBBoxes(contours));
    // }

    // // Filter the bounding boxes
    // vector<vector<RotatedRect>> filteredBBoxesVector;
    // for (auto& bboxes : bboxesVector) {
    //     // filteredBBoxesVector.push_back(segmenter.filterBBoxes(bboxes));
    // }

    // // // Draw the bounding boxes
    // // for (size_t i = 0; i < imgVector.size(); ++i) {
    // //     Mat img = imgVector[i];
    // //     const auto& bboxes = bboxesVector[i];
    // //     const auto& bboxes = filteredBBoxesVector[i];
    // //     segmenter.drawBBoxes(img, bboxes);
    // //     segmenter.showImages(img);
    // // }
    
    // // Segment the cars
    // vector<Mat> segmentedCarsVector;
    // for (size_t i = 0; i < imgVector.size(); ++i) {
    //     const auto& bboxes = bboxesVector[i];
    //     const auto& mask = enhancedMaskVector[i];
    //     Mat img = Mat::zeros(imgVector[i].size(), CV_8UC3);
    //     segmentedCarsVector.push_back(segmenter.segmentCar(bboxes, groundtruthBBoxes, mask, img));
    // }

    // // Create the mask for the evaluation
    // vector<Mat> segmentedMaskVector;
    // for (const auto& img : segmentedCarsVector) {
    //     segmentedMaskVector.push_back(segmenter.createSegmentMask(img));
    // }

    // // Show the images
    // for (const auto& img: segmentedMaskVector) {
    //     // segmenter.showImages(img);
    // }

    // // Evaluate the segmentation
    // mIoU miou;
    // vector<double> miouVectorForImg;
    // vector<int> classes = {0, 1, 2};
    // for (size_t i = 0; i < segmentedMaskVector.size(); i++) 
    //     miouVectorForImg.push_back(miou.computeMeanIoU(evaluationMaskVector[i], segmentedMaskVector[i], classes));

    // // Compute the mean IoU for each sequence
    // vector<double> miouVectorForSequence;
    // for (size_t i = 0; i < sequences.size(); i++) {
    //     double sum = 0;
    //     for (size_t j = 0; j < evaluationMaskVector.size(); j++) {
    //         if (j / 5 == i) sum += miouVectorForImg[j];
    //     }
    //     miouVectorForSequence.push_back(sum / 5);
    //     cout << "Mean IoU for sequence " << (i + 1) << ": " << sum / 5 << endl;
    // }

    return 0;
}
