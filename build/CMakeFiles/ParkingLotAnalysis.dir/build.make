# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/pietro/Analysis-of-parking-lot-occupancy

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/pietro/Analysis-of-parking-lot-occupancy/build

# Include any dependencies generated for this target.
include CMakeFiles/ParkingLotAnalysis.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/ParkingLotAnalysis.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/ParkingLotAnalysis.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ParkingLotAnalysis.dir/flags.make

CMakeFiles/ParkingLotAnalysis.dir/src/main.cpp.o: CMakeFiles/ParkingLotAnalysis.dir/flags.make
CMakeFiles/ParkingLotAnalysis.dir/src/main.cpp.o: ../src/main.cpp
CMakeFiles/ParkingLotAnalysis.dir/src/main.cpp.o: CMakeFiles/ParkingLotAnalysis.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pietro/Analysis-of-parking-lot-occupancy/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/ParkingLotAnalysis.dir/src/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ParkingLotAnalysis.dir/src/main.cpp.o -MF CMakeFiles/ParkingLotAnalysis.dir/src/main.cpp.o.d -o CMakeFiles/ParkingLotAnalysis.dir/src/main.cpp.o -c /home/pietro/Analysis-of-parking-lot-occupancy/src/main.cpp

CMakeFiles/ParkingLotAnalysis.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ParkingLotAnalysis.dir/src/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pietro/Analysis-of-parking-lot-occupancy/src/main.cpp > CMakeFiles/ParkingLotAnalysis.dir/src/main.cpp.i

CMakeFiles/ParkingLotAnalysis.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ParkingLotAnalysis.dir/src/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pietro/Analysis-of-parking-lot-occupancy/src/main.cpp -o CMakeFiles/ParkingLotAnalysis.dir/src/main.cpp.s

CMakeFiles/ParkingLotAnalysis.dir/src/ParkingSpaceDetector.cpp.o: CMakeFiles/ParkingLotAnalysis.dir/flags.make
CMakeFiles/ParkingLotAnalysis.dir/src/ParkingSpaceDetector.cpp.o: ../src/ParkingSpaceDetector.cpp
CMakeFiles/ParkingLotAnalysis.dir/src/ParkingSpaceDetector.cpp.o: CMakeFiles/ParkingLotAnalysis.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pietro/Analysis-of-parking-lot-occupancy/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/ParkingLotAnalysis.dir/src/ParkingSpaceDetector.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ParkingLotAnalysis.dir/src/ParkingSpaceDetector.cpp.o -MF CMakeFiles/ParkingLotAnalysis.dir/src/ParkingSpaceDetector.cpp.o.d -o CMakeFiles/ParkingLotAnalysis.dir/src/ParkingSpaceDetector.cpp.o -c /home/pietro/Analysis-of-parking-lot-occupancy/src/ParkingSpaceDetector.cpp

CMakeFiles/ParkingLotAnalysis.dir/src/ParkingSpaceDetector.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ParkingLotAnalysis.dir/src/ParkingSpaceDetector.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pietro/Analysis-of-parking-lot-occupancy/src/ParkingSpaceDetector.cpp > CMakeFiles/ParkingLotAnalysis.dir/src/ParkingSpaceDetector.cpp.i

CMakeFiles/ParkingLotAnalysis.dir/src/ParkingSpaceDetector.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ParkingLotAnalysis.dir/src/ParkingSpaceDetector.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pietro/Analysis-of-parking-lot-occupancy/src/ParkingSpaceDetector.cpp -o CMakeFiles/ParkingLotAnalysis.dir/src/ParkingSpaceDetector.cpp.s

CMakeFiles/ParkingLotAnalysis.dir/src/CarSegmenter.cpp.o: CMakeFiles/ParkingLotAnalysis.dir/flags.make
CMakeFiles/ParkingLotAnalysis.dir/src/CarSegmenter.cpp.o: ../src/CarSegmenter.cpp
CMakeFiles/ParkingLotAnalysis.dir/src/CarSegmenter.cpp.o: CMakeFiles/ParkingLotAnalysis.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pietro/Analysis-of-parking-lot-occupancy/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/ParkingLotAnalysis.dir/src/CarSegmenter.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ParkingLotAnalysis.dir/src/CarSegmenter.cpp.o -MF CMakeFiles/ParkingLotAnalysis.dir/src/CarSegmenter.cpp.o.d -o CMakeFiles/ParkingLotAnalysis.dir/src/CarSegmenter.cpp.o -c /home/pietro/Analysis-of-parking-lot-occupancy/src/CarSegmenter.cpp

CMakeFiles/ParkingLotAnalysis.dir/src/CarSegmenter.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ParkingLotAnalysis.dir/src/CarSegmenter.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pietro/Analysis-of-parking-lot-occupancy/src/CarSegmenter.cpp > CMakeFiles/ParkingLotAnalysis.dir/src/CarSegmenter.cpp.i

CMakeFiles/ParkingLotAnalysis.dir/src/CarSegmenter.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ParkingLotAnalysis.dir/src/CarSegmenter.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pietro/Analysis-of-parking-lot-occupancy/src/CarSegmenter.cpp -o CMakeFiles/ParkingLotAnalysis.dir/src/CarSegmenter.cpp.s

# Object files for target ParkingLotAnalysis
ParkingLotAnalysis_OBJECTS = \
"CMakeFiles/ParkingLotAnalysis.dir/src/main.cpp.o" \
"CMakeFiles/ParkingLotAnalysis.dir/src/ParkingSpaceDetector.cpp.o" \
"CMakeFiles/ParkingLotAnalysis.dir/src/CarSegmenter.cpp.o"

# External object files for target ParkingLotAnalysis
ParkingLotAnalysis_EXTERNAL_OBJECTS =

ParkingLotAnalysis: CMakeFiles/ParkingLotAnalysis.dir/src/main.cpp.o
ParkingLotAnalysis: CMakeFiles/ParkingLotAnalysis.dir/src/ParkingSpaceDetector.cpp.o
ParkingLotAnalysis: CMakeFiles/ParkingLotAnalysis.dir/src/CarSegmenter.cpp.o
ParkingLotAnalysis: CMakeFiles/ParkingLotAnalysis.dir/build.make
ParkingLotAnalysis: /usr/local/lib/libopencv_gapi.so.4.9.0
ParkingLotAnalysis: /usr/local/lib/libopencv_highgui.so.4.9.0
ParkingLotAnalysis: /usr/local/lib/libopencv_ml.so.4.9.0
ParkingLotAnalysis: /usr/local/lib/libopencv_objdetect.so.4.9.0
ParkingLotAnalysis: /usr/local/lib/libopencv_photo.so.4.9.0
ParkingLotAnalysis: /usr/local/lib/libopencv_stitching.so.4.9.0
ParkingLotAnalysis: /usr/local/lib/libopencv_video.so.4.9.0
ParkingLotAnalysis: /usr/local/lib/libopencv_videoio.so.4.9.0
ParkingLotAnalysis: /usr/local/lib/libopencv_imgcodecs.so.4.9.0
ParkingLotAnalysis: /usr/local/lib/libopencv_dnn.so.4.9.0
ParkingLotAnalysis: /usr/local/lib/libopencv_calib3d.so.4.9.0
ParkingLotAnalysis: /usr/local/lib/libopencv_features2d.so.4.9.0
ParkingLotAnalysis: /usr/local/lib/libopencv_flann.so.4.9.0
ParkingLotAnalysis: /usr/local/lib/libopencv_imgproc.so.4.9.0
ParkingLotAnalysis: /usr/local/lib/libopencv_core.so.4.9.0
ParkingLotAnalysis: CMakeFiles/ParkingLotAnalysis.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/pietro/Analysis-of-parking-lot-occupancy/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable ParkingLotAnalysis"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ParkingLotAnalysis.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ParkingLotAnalysis.dir/build: ParkingLotAnalysis
.PHONY : CMakeFiles/ParkingLotAnalysis.dir/build

CMakeFiles/ParkingLotAnalysis.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ParkingLotAnalysis.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ParkingLotAnalysis.dir/clean

CMakeFiles/ParkingLotAnalysis.dir/depend:
	cd /home/pietro/Analysis-of-parking-lot-occupancy/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/pietro/Analysis-of-parking-lot-occupancy /home/pietro/Analysis-of-parking-lot-occupancy /home/pietro/Analysis-of-parking-lot-occupancy/build /home/pietro/Analysis-of-parking-lot-occupancy/build /home/pietro/Analysis-of-parking-lot-occupancy/build/CMakeFiles/ParkingLotAnalysis.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ParkingLotAnalysis.dir/depend

