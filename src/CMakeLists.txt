include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

add_executable(mnist mnist.cpp)
add_executable(rtgpp main.cpp)
target_link_libraries(rtgpp "${TORCH_LIBRARIES}")
target_link_libraries(mnist "${TORCH_LIBRARIES}")

add_subdirectory(./nmt/)
