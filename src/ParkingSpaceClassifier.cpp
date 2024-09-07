#include "ParkingSpaceClassifier.h"

using namespace cv;

// Function to create a bounding box
Mat createBoundingBox(const Mat &parkingLotImage, const RotatedRect &rotated_rect) {
    Mat rotation_matrix = getRotationMatrix2D(rotated_rect.center, rotated_rect.angle, 1.0);
        // Rotate the entire image around the bounding box center
        Mat rotated_image;
        warpAffine(parkingLotImage, rotated_image, rotation_matrix, parkingLotImage.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0,0,0));
        
        // Extract the ROI from the rotated image
        Size rect_size = rotated_rect.size;
        
        // Create a 3x3 full rotation matrix from the 2x3 rotation matrix
        cv::Rect bounding_box(
            static_cast<int>(rotated_rect.center.x - rect_size.width / 2),
            static_cast<int>(rotated_rect.center.y - rect_size.height / 2),
            static_cast<int>(rect_size.width),
            static_cast<int>(rect_size.height)
        );

        // Ensure the bounding box is within the image bounds
        bounding_box &= cv::Rect(0, 0, rotated_image.cols, rotated_image.rows);
        Mat cropped_image;

        if (bounding_box.width > 0 && bounding_box.height > 0) {
            cropped_image = rotated_image(bounding_box);
        }
    return cropped_image;
}

void classifyParkingSpaces(const Mat &parkingLotImage, std::vector<RotatedRect> &parkingSpaces, std::vector<bool> &occupancyStatus) {
    // Convert the image from BGR to HSV color space
    Mat image = parkingLotImage;
    cv::Mat hsvImage;
    cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);

    // Define the range of the color you want to delete (in HSV space)
    // For example, if you want to delete a green color
    cv::Scalar lowerBound(0, 0, 10); // Lower bound of HSV for green
    cv::Scalar upperBound(179, 50, 100); // Upper bound of HSV for green

    // Create a mask where the specified color range is white (255) and the rest is black (0)
    cv::Mat mask;
    cv::inRange(hsvImage, lowerBound, upperBound, mask);

    // Optionally, you can dilate or erode the mask to refine it
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::dilate(mask, mask, kernel);

    // Invert the mask to keep the areas outside the color range
    cv::Mat maskInv;
    cv::bitwise_not(mask, maskInv);

    // Use the mask to delete the specific color (set it to black or another color)
    cv::Mat result;
    cv::bitwise_and(image, image, result, maskInv);
    cv::imshow("Original Image", image);
    cv::imshow("Mask", mask);
    cv::imshow("Result Image", result);
    cv::waitKey(0);

    for (size_t i = 0; i < parkingSpaces.size(); ++i) {
        // Get the rotation matrix for the bounding box
        const RotatedRect &rotated_rect = parkingSpaces[i];
        
        Mat cropped_bbox = createBoundingBox(parkingLotImage, rotated_rect);

        // Convert to grayscale
        Mat grayRoi;
        cvtColor(cropped_bbox, grayRoi, COLOR_BGR2GRAY);

        cv::Mat blurred_image;
        cv::GaussianBlur(grayRoi, blurred_image, cv::Size(5, 5), 0);

        // Perform contrast stretching using cv::normalize
        cv::Mat stretched;
        equalizeHist( grayRoi, stretched );

        // Apply Canny edge detection
        Mat edges;
        Canny(stretched, edges, 70, 300, 3, true);
        
        // Count the number of edge pixels (non-zero pixels in the edge image)
        double edgeCount = countNonZero(stretched);

        // Apply Otsu's method to find the optimal threshold
        double otsuThreshVal = cv::threshold(edges, edges, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        
        // Heuristic: If there are many non-zero pixels, the space is occupied; otherwise, it's empty
        if (edgeCount > 0.6 * stretched.total()) {  // Adjust the threshold as necessary
            occupancyStatus[i] = true;
        } else {
            occupancyStatus[i] = false;
        }
        std::cerr << occupancyStatus[i] << "," << edgeCount / stretched.total() << "," << otsuThreshVal << std::endl;
    }

}

// Function to draw the parking spaces on the image
void drawParkingSpaces(Mat &image, const std::vector<RotatedRect> &parkingSpaces, const std::vector<bool> &occupancyStatus) {
    for (size_t i = 0; i < parkingSpaces.size(); ++i) {
        Point2f vertices[4];
        parkingSpaces[i].points(vertices);
        
        Scalar color = occupancyStatus[i] ? Scalar(0, 0, 255) : Scalar(0, 255, 0);

        for (int j = 0; j < 4; j++)
            line(image, vertices[j], vertices[(j+1)%4], color, 2);
    }
}
