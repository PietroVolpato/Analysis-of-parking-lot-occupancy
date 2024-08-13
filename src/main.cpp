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
    std::vector<Mat> edges = detectEdges(images);

    // Detect the lines in the images
    std::vector<std::vector<Vec4i>> lines = detectLines(edges);

   // Draw the lines in the images
    for (int i = 0; i < imgs.size(); i++) {
        for (int j = 0; j < lines[i].size(); j++) {
            line(imgs[i], Point(lines[i][j][0], lines[i][j][1]), Point(lines[i][j][2], lines[i][j][3]), Scalar(0, 0, 255), 3, LINE_AA);
        }
    }

    // // Draw the bounding boxes in the images
    // for (int i = 0; i < imgs.size(); i++ ) {
    //     drawParkingSpaces(imgs[i], lines[i], contours[i]);
    // }

    // std::vector<std::vector<std::vector<Vec4i>>> parallelLines;
    // for (int i = 0; i < lines.size(); i++) {
    //     parallelLines.push_back(filterLines(lines[i]));
    // }

    // for (int i = 0; i < imgs.size(); i++) {
    //     for (int j = 0; j < parallelLines[i].size(); j++) {
    //         for (int k = 0; k < parallelLines[i][j].size(); k++) {
    //             line(imgs[i], Point(parallelLines[i][j][k][0], parallelLines[i][j][k][1]), Point(parallelLines[i][j][k][2], parallelLines[i][j][k][3]), Scalar(0, 0, 255), 3, LINE_AA);
    //         }
    //     }

    //     // drawBoundingBoxes(imgs[i], parallelLines[i], contours[i]);
    // }

    // // Detect the bounding boxes in the images
    // for (int i = 0; i < imgs.size(); i++) {
    //     drawBoundingBoxes(imgs[i], lines[i]);
    // }

    // Show the images
    showImages(images);

    return 0;
}