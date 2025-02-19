#include "Visualizer.h"

using namespace std;
using namespace cv;

// Constructor
Visualizer::Visualizer(int width , int height): minimap_width(width), minimap_height(height) {} // Default values , Constructor


void Visualizer::drawParkingSpaces(Mat &image, const vector<RotatedRect> &parkingSpaces, const vector<bool> &occupancyStatus) {
    for (size_t i = 0; i < parkingSpaces.size(); ++i) {
        Point2f vertices[4];
        parkingSpaces[i].points(vertices);
        
        // Set color based on occupancy status (Red for occupied, Green for free)
        Scalar color = occupancyStatus[i] ? Scalar(0, 0, 255) : Scalar(0, 255, 0);

        for (int j = 0; j < 4; j++) {
            line(image, vertices[j], vertices[(j + 1) % 4], color, 2);
        }
    }
}

void Visualizer::drawRotatedRect(Mat& image, float center_x, float center_y, float angle, Scalar color) {
    RotatedRect rrect(Point2f(center_x, center_y), Size2f(Visualizer::spaceWidth, Visualizer::spaceHeight), angle);
    Point2f vertices[4];
    rrect.points(vertices);

    // Convert to a polygon and fill it
    vector<Point> contour(vertices, vertices + 4);
    fillPoly(image, vector<vector<Point>>{contour}, color);

    for (int i = 0; i < 4; i++) {
        line(image, vertices[i], vertices[(i + 1) % 4], Scalar(0, 0, 0), 1);
    }
}

void Visualizer::drawParkingRow(Mat& image, int numSpaces, float startX, float startY, float angle, vector<bool>& occupancy, int& index) {
    float cx, cy = startY;
    
    for (int i = 0; i < numSpaces; i++) {
        cx = startX - i * Visualizer::deltaX;
        
        // Assign color based on occupancy vector
        bool occupied = occupancy[index++];
        Scalar color = occupied ? Scalar(0, 0, 255) : Scalar(255, 0, 0); // Red for occupied, Blue for free

        drawRotatedRect(image, cx, cy, angle, color);
    }
}

Mat Visualizer::overlaySmallOnLarge(const Mat& parkingSapaceClassified, const Mat& visual2D) {

    // Clone the large image to keep the original unchanged
    Mat outputImage = parkingSapaceClassified.clone();

    // The size for the small image (1/4 width of the large image)
    int newWidth = parkingSapaceClassified.cols / 4;
    int newHeight = (visual2D.rows * newWidth) / visual2D.cols;  // Maintain aspect ratio

   // Resize the small image
   Mat resizedSmallImage;
   resize(visual2D, resizedSmallImage, cv::Size(newWidth, newHeight));

   // Define the ROI (bottom-left corner)
   Rect roi(0, outputImage.rows - newHeight, newWidth, newHeight);

   // Copy small image into the ROI
   resizedSmallImage.copyTo(outputImage(roi));

   return outputImage;
}



Mat Visualizer::createMockMinimap(vector<bool> &occupancy) {
    Mat minimap(Visualizer::minimap_height, Visualizer::minimap_width, CV_8UC3, Scalar(255, 255, 255));

    int index = 0; // Track occupancy vector position
    drawParkingRow(minimap, 9, Visualizer::startX,  Visualizer::startY, -45.0f, occupancy, index);
    drawParkingRow(minimap, 9, Visualizer::startX + 10,  Visualizer::startY + Visualizer::deltaY, 45.0f, occupancy, index);
    drawParkingRow(minimap, 10, Visualizer::startX,  Visualizer::startY + 3 * Visualizer::deltaY, -45.0f, occupancy, index);
    drawParkingRow(minimap, 10, Visualizer::startX,  Visualizer::startY + 4 * Visualizer::deltaY, -45.0f, occupancy, index);

    return minimap;
}
