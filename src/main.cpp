#include "ParkingSpaceDetector.h"
/*#include "ParkingSpaceClassifier.h"
#include "CarSegmenter.h"
#include "Visualizer.h"*/

using namespace cv;

int main (int argc, char** argv) {
    // Load the images
    int sequence = 0;
    std::vector<Mat> imgs = loadImages(sequence);

    // Preprocess the images
    std::vector<Mat> images = preprocessImages(imgs);

    // Detect the edges in the images
    std::vector<Mat> edges = detectEdges(images, 50, 150);

    // Detect the lines in the images
    std::vector<std::vector<Vec4i>> lines = detectLines(edges, 50, 50, 10);

    // Filter the lines
    std::vector<std::vector<Vec4i>> filteredLines = filterLines(lines);

    // Draw the lines
    drawLines(imgs, filteredLines);

    // Draw the bounding boxes
    // drawBoundingBoxes(imgs, lines);

    // Show the images
    showImages(imgs);

    return 0;
}