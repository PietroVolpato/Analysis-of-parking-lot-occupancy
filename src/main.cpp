#include "ParkingSpaceDetector.h"
// #include "ParkingSpaceClassifier.h"
// #include "CarSegmenter.h"
// #include "Visualizer.h"

using namespace cv;

int main (int argc, char** argv) {
    // Load the images
    int sequence = 0;
    std::vector<Mat> imgs = loadImages(sequence);

    // Preprocess the images
    // std::vector<Mat> images = preprocessImages(imgs);

    std::vector<Mat> imgVector;
    for (const auto& img: imgs) {
        Mat binaryImg;
        threshold(img, binaryImg, 200, 255, THRESH_BINARY);
        Mat grayImg;
        cvtColor(binaryImg, grayImg, COLOR_BGR2GRAY);
        Mat dilatedImg;
        dilate(grayImg, dilatedImg, Mat(), Point(-1, -1), 2);
        imgVector.push_back(dilatedImg);
    }

    // Detect the edges in the images
    // std::vector<Mat> edges = detectEdges(images, 50, 150);

    // Detect the lines in the images
    std::vector<std::vector<Vec4i>> lines = detectLines(imgVector, 100, 50, 10);

    // Filter the lines
    // std::vector<std::vector<Vec4i>> filteredLines = filterLines(lines);

    // Draw the lines
    drawLines(imgs, lines); 

    // Draw the bounding boxes
    // drawBoundingBoxes(imgs, filteredLines);

    // Show the images
    showImages(imgs);

    return 0;
}