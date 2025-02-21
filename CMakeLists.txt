cmake_minimum_required(VERSION 3.10)

# Set the project name and version
project(ParkingLotAnalysis)

# Add OpenCV package
find_package(OpenCV REQUIRED)
find_package(pugixml REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories((${PROJECT_SOURCE_DIR}/include))

# Add the executable
add_executable(${PROJECT_NAME} 
    src/main.cpp 
    src/tinyxml2.cpp
    src/GroundTruthReader.cpp
    src/ParkingSpaceDetector.cpp 
    # src/ParkingSpaceClassifier.cpp 
    # src/CarSegmenter.cpp 
    # src/Visualizer.cpp
)

# Link libraries
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} pugixml::pugixml)
