#include "Visualizer.h"

using namespace std;
using namespace cv;

// Constructor
Visualizer::Visualizer() {}

void Visualizer::drawRotatedRect(Mat& image, const float &center_x, const float &center_y, const Scalar &color, const bool &occupancyStatus) {
    RotatedRect rrect(Point2f(center_x, center_y), Size2f(Visualizer::spaceWidth, Visualizer::spaceHeight), Visualizer::angle);
    Point2f vertices[4];
    rrect.points(vertices);
    
    // Convert to a polygon and fill it
    vector<Point> contour(vertices, vertices + 4);
    fillPoly(image, vector<vector<Point>>{contour}, color);
    
    for (int i = 0; i < 4; i++) {
        line(image, vertices[i], vertices[(i + 1) % 4], Scalar(0, 0, 0), 1);
    }
}

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

Mat Visualizer::createMockMinimap(int width, int height) {
    Mat minimap(height, width, CV_8UC3, Scalar(255, 255, 255));

    // Parking space parameters
    int numSpaces;
    float cx, cy;

    // Row 1
    numSpaces = 8;
    cy = Visualizer::startY;
    for (int i = 0; i < numSpaces; i++) {
        cx = Visualizer::startX + i * deltaX;
        drawRotatedRect(minimap, cx, cy, Scalar(0, 0, 255), rand() % 2); // Red
    }

    // Row 2
    angle = 45.0f;
    numSpaces = 9;
    float startX2 = startX + 10;
    startY += deltaY;
    cy = startY;
    for (int i = 0; i < numSpaces; i++) {
        cx = startX2 + i * deltaX;
        RotatedRect rrect(Point2f(cx, cy), Size2f(spaceWidth, spaceHeight), angle);
        Scalar color = (i % 2 == 0) ? Scalar(255, 0, 0) : Scalar(0, 0, 255); // Blue & Red alternating
        drawRotatedRect(minimap, cx, cy, color, rand() % 2);
    }

    // Row 3
    angle = -45.0f;
    numSpaces = 10;
    startY = 200.0f;
    cy = startY;
    for (int i = 0; i < numSpaces; i++) {
        cx = startX + i * deltaX;
        RotatedRect rrect(Point2f(cx, cy), Size2f(spaceWidth, spaceHeight), angle);
        bool occupied = ((i + 1) % 3 == 0);
        Scalar color = occupied ? Scalar(0, 0, 255) : Scalar(255, 0, 0);
        drawRotatedRect(minimap, cx, cy, color, rand() % 2);
    }

    // Row 4
    numSpaces = 10;
    startY += deltaY;
    cy = startY;
    for (int i = 0; i < numSpaces; i++) {
        cx = startX + i * deltaX;
        RotatedRect rrect(Point2f(cx, cy), Size2f(spaceWidth, spaceHeight), angle);
        bool occupied = ((i + 1) % 3 == 0);
        Scalar color = occupied ? Scalar(0, 0, 255) : Scalar(255, 0, 0);
        drawRotatedRect(minimap, cx, cy, color, rand() % 2);
    }

    return minimap;
}
