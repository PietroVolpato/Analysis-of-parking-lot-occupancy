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

/**
 * @brief Draws a rotated rectangle (parking space) onto the image.
 * @param image  The image on which to draw.
 * @param rrect  The cv::RotatedRect representing the space.
 * @param color  The color (BGR) to use.
 */
void drawRotatedRect(cv::Mat& image, const cv::RotatedRect& rrect, const cv::Scalar& color) {
    cv::Point2f vertices[4];
    rrect.points(vertices);
    for (int i = 0; i < 4; i++) {
        cv::line(image, vertices[i], vertices[(i + 1) % 4], color, 2);
    }
}

/**
 * @brief Creates a mock parking lot layout with diagonal/zigzag rows of spaces.
 *        Some are colored red (occupied), others blue (free).
 * @param width   Width of the output image.
 * @param height  Height of the output image.
 * @return        The generated minimap as a cv::Mat.
 */
cv::Mat createMockMinimap(int width, int height) {
    // Start with a white background
    cv::Mat minimap(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

    // Parameters to tweak the appearance
    float spaceWidth  = 40.0f;  // width of each parking space
    float spaceHeight = 20.0f;  // height of each parking space
    float angle       = - 45.0f; // rotation angle (negative = slanted left)

    // We'll define rows of parking spaces at different Y positions
    // and shift each space slightly in X and Y to create a zigzag effect.
    
    // ------------------------
    // Row 1: 6 spaces, all "occupied" (red)
    // ------------------------
    {
        int numSpaces = 8;
        float startX = 70.0f;  // starting X
        float startY = 60.0f;  // row Y
        float deltaX = spaceHeight / cos(angle * CV_PI / 180.0));  // how much we move right for each next space
        float deltaY = 0;   // how much we move down for each next space

        for (int i = 0; i < numSpaces; i++) {
            // Center for this space
            float cx = startX + i * deltaX;
            float cy = startY + i * deltaY;

            cv::RotatedRect rrect(cv::Point2f(cx, cy),
                                   cv::Size2f(spaceWidth, spaceHeight),
                                   angle);

            // Red in BGR
            cv::Scalar color(0, 0, 255); 
            drawRotatedRect(minimap, rrect, color);
        }
    }

    // ------------------------
    // Row 2: 8 spaces, mix of blue (free) and red (occupied)
    // ------------------------
    {
        int numSpaces = 8;
        float startX = 50.0f;  // starting X
        float startY = 140.0f; // row Y
        float deltaX = 35.0f;
        float deltaY = 5.0f;

        // Let’s alternate colors: free, occupied, free, occupied, ...
        // Blue (BGR) = (255, 0, 0)
        // Red  (BGR) = (0, 0, 255)
        for (int i = 0; i < numSpaces; i++) {
            float cx = startX + i * deltaX;
            float cy = startY + i * deltaY;

            cv::RotatedRect rrect(cv::Point2f(cx, cy),
                                   cv::Size2f(spaceWidth, spaceHeight),
                                   angle);

            cv::Scalar color = (i % 2 == 0) ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 0, 255);
            drawRotatedRect(minimap, rrect, color);
        }
    }

    // ------------------------
    // Row 3: 8 spaces, mostly free (blue) with a few occupied (red)
    // ------------------------
    {
        int numSpaces = 8;
        float startX = 70.0f;   // starting X
        float startY = 220.0f;  // row Y
        float deltaX = 35.0f;
        float deltaY = 5.0f;

        // Example: Mark every 3rd space as occupied
        for (int i = 0; i < numSpaces; i++) {
            float cx = startX + i * deltaX;
            float cy = startY + i * deltaY;

            cv::RotatedRect rrect(cv::Point2f(cx, cy),
                                   cv::Size2f(spaceWidth, spaceHeight),
                                   angle);

            bool occupied = ((i + 1) % 3 == 0); 
            cv::Scalar color = occupied ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0);
            drawRotatedRect(minimap, rrect, color);
        }
    }

    return minimap;
}

void showMinimap(const cv::Mat& minimap, const std::string& windowName) {
    cv::imshow(windowName, minimap);
    cv::waitKey(1);
}
