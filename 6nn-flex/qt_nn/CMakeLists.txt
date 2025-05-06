cmake_minimum_required(VERSION 3.10)
project(WaterPotabilityNN_Qt)

# Automatically include current directory
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Find includes in corresponding build directories
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# Find Qt5 packages
find_package(Qt5 COMPONENTS Core Widgets Gui REQUIRED)
find_package(Qt5Test QUIET)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
)

# QCustomPlot library
add_library(qcustomplot
    qcustomplot/qcustomplot.cpp
)
target_link_libraries(qcustomplot
    Qt5::Core
    Qt5::Widgets
    Qt5::Gui
)

# Generate the executable
if(WIN32)
    add_executable(water_potability_nn_qt WIN32
        src/qt_bindings.cpp
        src/qt_main.cpp
    )
else()
    add_executable(water_potability_nn_qt
        src/qt_bindings.cpp
        src/qt_main.cpp
    )
endif()

# Link libraries
target_link_libraries(water_potability_nn_qt
    Qt5::Core
    Qt5::Widgets
    Qt5::Gui
    qcustomplot
)

# Installation
install(TARGETS water_potability_nn_qt
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Testing if Qt5Test was found
if(Qt5Test_FOUND)
    enable_testing()
    add_subdirectory(tests)
endif() 