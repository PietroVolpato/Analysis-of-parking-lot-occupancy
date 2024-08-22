#include "ParkingSpaceClassifier.h"
#include <opencv2/opencv.hpp>
#include "tinyxml2.h"
#include <string>

void showBoundingBoxesSeparately(const cv::Mat &parkingLotImage, const std::vector<cv::RotatedRect> &parkingSpaces) {
    const double scaleFactor = 4.0;
    
    for (const auto& rotated_rect : parkingSpaces) {
        // Step 1: Get the rotation matrix
        cv::Mat rotation_matrix = cv::getRotationMatrix2D(rotated_rect.center, rotated_rect.angle, 1.0);

        // Step 2: Apply the affine transformation to rotate the entire image
        cv::Mat rotated_image;
        cv::warpAffine(parkingLotImage, rotated_image, rotation_matrix, parkingLotImage.size());

        // Step 3: Crop the aligned rectangle
        cv::Size rect_size = rotated_rect.size;
        cv::Rect bounding_box(
            static_cast<int>(rotated_rect.center.x - rect_size.width / 2),
            static_cast<int>(rotated_rect.center.y - rect_size.height / 2),
            static_cast<int>(rect_size.width),
            static_cast<int>(rect_size.height)
        );

        // Ensure the bounding box is within the image bounds
        bounding_box &= cv::Rect(0, 0, rotated_image.cols, rotated_image.rows);

        if (bounding_box.width > 0 && bounding_box.height > 0) {
            cv::Mat cropped_image = rotated_image(bounding_box);

            // Scale up the cropped image
            cv::Mat scaled_image;
            cv::resize(cropped_image, scaled_image, cv::Size(), scaleFactor, scaleFactor, cv::INTER_LINEAR);

            // Display the cropped and scaled image
            std::string window_name = "Cropped Image " + std::to_string(&rotated_rect - &parkingSpaces[0]);
            cv::imshow(window_name, scaled_image);
        }
    }
    cv::waitKey(0);  // Wait for a key press to close all windows
}

int main() {
    // Load the parking lot image
    cv::Mat parkingLotImage = cv::imread("data/sequence1/frames/2013-02-22_07_10_01.png");
    if (parkingLotImage.empty()) {
        std::cerr << "Error: Unable to load image!" << std::endl;
        return -1;
    }

    // Clone the image to create a separate copy for each method of occupancy detection
    cv::Mat imageFromXML = parkingLotImage.clone();
    cv::Mat imageFromDetection = parkingLotImage.clone();

    // Path to the XML file
    std::string xmlFilePath = "data/sequence1/bounding_boxes/2013-02-22_07_10_01.xml";

    // 1. Draw the parking spaces based on the XML file (using the occupancy status from the XML file)
    drawTrueParkingSpaces(imageFromXML, xmlFilePath);

    // 2. Draw the parking spaces based on the occupancy detected using the isOccupied function
    std::vector<bool> occupancyStatus;
    std::vector<cv::RotatedRect> parkingSpaces = extractBoundingBoxesFromXML(xmlFilePath, occupancyStatus);

    // Show bounding boxes separately
    // showBoundingBoxesSeparately(parkingLotImage, parkingSpaces);
    
    classifyParkingSpaces(parkingLotImage, parkingSpaces, occupancyStatus);  // Re-classify using K-Means
    drawParkingSpaces(imageFromDetection, parkingSpaces, occupancyStatus);

    // Determine the maximum width and height that can fit on the screen
    int screenHeight = 400;  // Example screen height, adjust as needed
    int maxImageHeight = std::min(imageFromXML.rows, screenHeight);
    
    // Resize images to fit within the screen height
    double scaleFactor = static_cast<double>(maxImageHeight) / imageFromXML.rows;

    cv::resize(imageFromXML, imageFromXML, cv::Size(), scaleFactor, scaleFactor);
    cv::resize(imageFromDetection, imageFromDetection, cv::Size(), scaleFactor, scaleFactor);

    // Combine the two images side by side for comparison
    cv::Mat combined;
    cv::hconcat(imageFromXML, imageFromDetection, combined);

    // Display the combined result
    cv::imshow("Parking Space Occupancy Comparison", combined);
    cv::waitKey(0);

    return 0;
}
