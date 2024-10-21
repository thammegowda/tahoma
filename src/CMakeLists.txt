
set(TAHOMA_SOURCES
    ./config.cpp
    ./data.cpp
    ./utils.cpp
    ./inference/decoder.cpp
    ./layer/transformer.cpp
    ./model/transformer_lm.cpp
    ./model/transformer_nmt.cpp
    ./train/criterion.cpp
    ./train/loss_computer.cpp
    ./train/stats_counter.cpp
    ./train/trainer.cpp
    ./train/utils.cpp
)


add_library(tahoma ${TAHOMA_SOURCES})
set_target_properties(tahoma PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_link_libraries(tahoma PUBLIC ${EXT_LIBS})
target_include_directories(tahoma PUBLIC "${CMAKE_SOURCE_DIR}/include")

add_executable(tahoma_exe main.cpp ${BACKWARD_ENABLE})
set_target_properties(tahoma_exe PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set_target_properties(tahoma_exe PROPERTIES OUTPUT_NAME "tahoma")
target_link_libraries(tahoma_exe PUBLIC tahoma)
add_backward(tahoma_exe)   # debug info if process crashes


#add_executable(mnist mnist.cpp ${BACKWARD_ENABLE})
#add_backward(mnist)   # debug info for crash
#set_target_properties(mnist PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
#target_link_libraries(mnist ${EXT_LIBS})

# add_executable(buffer buffer.cpp ${BACKWARD_ENABLE})
# add_backward(buffer)   # debug info for crash
# set_target_properties(buffer PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
# target_link_libraries(buffer ${EXT_LIBS})

