#include "ParkingSpaceDetector.h"
// #include "ParkingSpaceClassifier.h"
// #include "CarSegmenter.h"
// #include "Visualizer.h"

using namespace cv;

int main (int argc, char** argv) {
    // Load the images
    int sequence = 0;
    std::vector<Mat> imgVector = loadImages(sequence);

     // Contrast stretch the images
    std::vector<Mat> stretchedImgVector = constrastStretch(imgVector);

    // Preprocess the images
    std::vector<Mat> processedImgVector = preprocessImages(stretchedImgVector);

    // Detect the edges in the images
    std::vector<Mat> edges = detectEdges(processedImgVector, 50, 150);

    // Detect the lines in the images
    std::vector<std::vector<Vec4i>> lines = detectLines(edges, 50, 50, 20);

    // Filter the lines
    std::vector<std::vector<Vec4i>> filteredLines = filterLines(lines, 10, 30, 100, 200, 30, 100);

    // Draw the lines
    drawLines(imgVector, filteredLines); 
    // drawLines(imgVector, lines);

    // Draw the bounding boxes
    // drawBoundingBoxes(imgs, filteredLines);

    // Show the images
    showImages(imgVector);

    return 0;
}