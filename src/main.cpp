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
    std::vector<Mat> processedImgVector = preprocessImages(imgVector);
    

    // Detect the edges in the images
    std::vector<Mat> edges = detectEdges(processedImgVector, 300, 700);

    // Detect the lines in the images
    std::vector<std::vector<Vec4i>> lines = detectLines(processedImgVector, 110, 50, 10);

    // Filter the lines
    std::vector<std::vector<Vec4i>> filteredLines = filterLines(lines, 0, 50, 100, 200, 30, 100);

    // Draw the lines
    drawLines(imgVector, filteredLines); 
    // drawLines(imgVector, lines);

    // Draw the bounding boxes
    // drawBoundingBoxes(imgs, filteredLines);

    // Show the images
    showImages(imgVector);

    return 0;
}