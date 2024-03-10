include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

add_subdirectory(./common/)
add_subdirectory(./layer/)
add_subdirectory(./model/)
add_subdirectory(./train/)


add_executable(tahoma main.cpp ${BACKWARD_ENABLE})
add_backward(tahoma)   # debug info for crash
set_target_properties(tahoma PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_link_libraries(tahoma ${EXT_LIBS})


#add_executable(mnist mnist.cpp ${BACKWARD_ENABLE})
#add_backward(mnist)   # debug info for crash
#set_target_properties(mnist PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
#target_link_libraries(mnist ${EXT_LIBS})

# add_executable(buffer buffer.cpp ${BACKWARD_ENABLE})
# add_backward(buffer)   # debug info for crash
# set_target_properties(buffer PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
# target_link_libraries(buffer ${EXT_LIBS})

