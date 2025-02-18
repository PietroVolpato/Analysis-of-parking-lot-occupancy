#include "Visualizer.h"

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

// Function to draw the parking spaces on the image
void drawParkingSpaces(cv::Mat &image, const std::vector<cv::RotatedRect> &parkingSpaces, const std::vector<bool> &occupancyStatus) {
    for (size_t i = 0; i < parkingSpaces.size(); ++i) {
        cv::Point2f vertices[4];
        parkingSpaces[i].points(vertices);
        
        cv::Scalar color = occupancyStatus[i] ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);

        for (int j = 0; j < 4; j++)
            line(image, vertices[j], vertices[(j+1)%4], color, 2);
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
    float deltaX = - spaceHeight / static_cast<float>(cos(angle * CV_PI / 180.0)); 
    float deltaY = (spaceWidth + spaceHeight) * static_cast<float>(cos(angle * CV_PI / 180.0)); ;
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
