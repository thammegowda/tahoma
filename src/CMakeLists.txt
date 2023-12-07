include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

add_subdirectory(./common/)
add_subdirectory(./layer/)
add_subdirectory(./model/)
add_subdirectory(./train/)


add_executable(rtgp main.cpp ${BACKWARD_ENABLE})
add_backward(rtgp)   # debug info for crash
set_target_properties(rtgp PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_link_libraries(rtgp ${EXT_LIBS})
#target_link_libraries(mnist "${TORCH_LIBRARIES}")
