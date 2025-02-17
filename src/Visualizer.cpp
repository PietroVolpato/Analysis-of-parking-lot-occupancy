#include "Visualizer.h"

cv::Mat createMinimap(int width, int height) {
    cv::Mat minimap = cv::Mat::zeros(height, width, CV_8UC3);
    minimap.setTo(cv::Scalar(255, 255, 255)); // White background
    return minimap;
}

void drawParkingSpace(cv::Mat& minimap, const cv::RotatedRect& space, const cv::Scalar& color) {
    cv::Point2f vertices[4];
    space.points(vertices);
    // Draw lines connecting the vertices (with wrapping)
    for (int i = 0; i < 4; i++) {
        cv::line(minimap, vertices[i], vertices[(i + 1) % 4], color, 2);
    }
}

void updateMinimap(cv::Mat& minimap, const std::vector<cv::RotatedRect>& parkingSpaces, 
                   const std::vector<bool>& occupancy) {
    if (parkingSpaces.size() != occupancy.size()) {
        std::cerr << "Error: The number of parking spaces does not match the occupancy vector size." << std::endl;
        return;
    }

    // Reset the minimap to a white background.
    minimap.setTo(cv::Scalar(255, 255, 255));

    // For each parking space, draw it in green if occupied, blue if free.
    for (size_t i = 0; i < parkingSpaces.size(); ++i) {
        cv::Scalar color = occupancy[i] ? cv::Scalar(0, 255, 0) : cv::Scalar(255, 0, 0);
        drawParkingSpace(minimap, parkingSpaces[i], color);
    }
}


void drawRotatedRect(cv::Mat& image, const cv::RotatedRect& rrect, const cv::Scalar& color) {
    cv::Point2f vertices[4];
    rrect.points(vertices);
     // Convert to a polygon and fill it
     std::vector<cv::Point> contour(vertices, vertices + 4);
     cv::fillPoly(image, std::vector<std::vector<cv::Point>>{contour}, color);
     for (int i = 0; i < 4; i++) {
        cv::line(image, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 0, 0), 1);
     }
}

cv::Mat createMockMinimap(int width, int height) {
    // Start with a white background
    cv::Mat minimap(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

    // Parameters to tweak the appearance
    float spaceWidth  = 40.0f;  // width of each parking space
    float spaceHeight = 20.0f;  // height of each parking space
    float angle       = - 45.0f; // rotation angle (negative = slanted left)
    int numSpaces;
    float startX = 400.0f;  // starting X
    float startY = 60.0f;  // row Y
    float deltaX = - spaceHeight / cos(angle * CV_PI / 180.0); 
    float deltaY = (spaceWidth + spaceHeight) * cos(angle * CV_PI / 180.0); ;
    float cx;
    float cy;

    // Row 1
    numSpaces = 8;
    cy = startY;
    for (int i = 0; i < numSpaces; i++) {
        // Center for this space
        cx = startX + i * deltaX;
        cv::RotatedRect rrect(cv::Point2f(cx, cy),
                                cv::Size2f(spaceWidth, spaceHeight),
                                angle);

        // Red in BGR
        cv::Scalar color(0, 0, 255); 
        drawRotatedRect(minimap, rrect, color);
    }


    // Row 2

    angle = 45.0f;
    numSpaces = 9;
    float startX2 = startX + 10;
    startY = startY + deltaY; // row Y
    cy = startY;
    // Let’s alternate colors: free, occupied, free, occupied, ...
    // Blue (BGR) = (255, 0, 0)
    // Red  (BGR) = (0, 0, 255)
    for (int i = 0; i < numSpaces; i++) {
        cx = startX2 + i * deltaX;

        cv::RotatedRect rrect(cv::Point2f(cx, cy),
                                cv::Size2f(spaceWidth, spaceHeight),
                                angle);

        cv::Scalar color = (i % 2 == 0) ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 0, 255);
        drawRotatedRect(minimap, rrect, color);
    }
    

    // ------------------------
    // Row 3
    // ------------------------

    angle = - 45.0f;
    numSpaces = 10;
    startY = 200.0f;  // row Y
    cy = startY;    

    // Example: Mark every 3rd space as occupied
    for (int i = 0; i < numSpaces; i++) {
        cx = startX + i * deltaX;

        cv::RotatedRect rrect(cv::Point2f(cx, cy),
                                cv::Size2f(spaceWidth, spaceHeight),
                                angle);

        bool occupied = ((i + 1) % 3 == 0); 
        cv::Scalar color = occupied ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0);
        drawRotatedRect(minimap, rrect, color);
    }

     // ------------------------
    // Row 4
    // ------------------------

    numSpaces = 10;
    startY = startY + deltaY; // row Y
    cy = startY;    

    // Example: Mark every 3rd space as occupied
    for (int i = 0; i < numSpaces; i++) {
        cx = startX + i * deltaX;

        cv::RotatedRect rrect(cv::Point2f(cx, cy),
                                cv::Size2f(spaceWidth, spaceHeight),
                                angle);

        bool occupied = ((i + 1) % 3 == 0); 
        cv::Scalar color = occupied ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0);
        drawRotatedRect(minimap, rrect, color);
    }



    return minimap;
}

void showMinimap(const cv::Mat& minimap, const std::string& windowName) {
    cv::imshow(windowName, minimap);
    cv::waitKey(1);
}
