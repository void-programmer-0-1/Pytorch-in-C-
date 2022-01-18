
CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

project(LinearRegression)

set(CMAKE_PREFIX_PATH /home/santosh/Desktop/AI/Torch/TorchScript/C++/LinearRegression/libtorch)
                                                    # path to the libtorch folder

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(AI 
    main.cpp
)

target_link_libraries(AI "${TORCH_LIBRARIES}")
set_property(TARGET AI PROPERTY CXX_STANDARD 14)