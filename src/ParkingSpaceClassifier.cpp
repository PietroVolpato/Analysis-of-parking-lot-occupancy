#include "ParkingSpaceClassifier.h"

using namespace cv;

// Function to create a bounding box
RotatedRect createBoundingBox(const Point2f& center, const Size2f& size, float angle) {
    return RotatedRect(center, size, angle);
}

void classifyParkingSpaces(const Mat &parkingLotImage, std::vector<RotatedRect> &parkingSpaces, std::vector<bool> &occupancyStatus) {
    for (size_t i = 0; i < parkingSpaces.size(); ++i) {
        // Get the rotation matrix for the bounding box
        const RotatedRect &rotated_rect = parkingSpaces[i];
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

        // Convert to grayscale
        Mat grayRoi;
        cvtColor(cropped_image, grayRoi, COLOR_BGR2GRAY);

        // Perform contrast stretching using cv::normalize
        cv::Mat stretched;
        cv::normalize(grayRoi, stretched, 0, 255, cv::NORM_MINMAX);

        cv::Mat blurred_image;
        cv::GaussianBlur(stretched, blurred_image, cv::Size(5, 5), 0);
        
        // Apply adaptive thresholding
        Mat binary;
        threshold(blurred_image, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

        // Perform morphological operations to clean the image
        Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
        Mat cleaned_image;
        morphologyEx(binary, cleaned_image, MORPH_CLOSE, kernel);


        imshow("grayscale roi", binary);
        waitKey(0);

        // Apply Canny edge detection
        Mat edges;
        Canny(stretched, edges, 70, 300, 3, true);
        
        // Count the number of edge pixels (non-zero pixels in the edge image)
        double edgeCount = countNonZero(cleaned_image);

        // Apply Otsu's method to find the optimal threshold
        double otsuThreshVal = cv::threshold(edges, edges, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        
        // Heuristic: If there are many non-zero pixels, the space is occupied; otherwise, it's empty
        if (edgeCount > 0.8 * cleaned_image.total()) {  // Adjust the threshold as necessary
            occupancyStatus[i] = true;
        } else {
            occupancyStatus[i] = false;
        }
        std::cerr << occupancyStatus[i] << "," << edgeCount / cleaned_image.total() << "," << otsuThreshVal << std::endl;
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
