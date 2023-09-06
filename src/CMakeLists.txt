include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")


#add_executable(mnist mnist.cpp)
add_executable(rtgp main.cpp)
set_target_properties(rtgp PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_link_libraries(rtgp "${TORCH_LIBRARIES}")
#target_link_libraries(mnist "${TORCH_LIBRARIES}")

add_subdirectory(./common/)
add_subdirectory(./nmt/)
