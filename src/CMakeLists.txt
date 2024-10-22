set(TAHOMA_SOURCES
    ./config.cpp
    ./data.cpp
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


add_library(tahoma_lib STATIC ${TAHOMA_SOURCES})
set_target_properties(tahoma_lib PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    OUTPUT_NAME "tahoma")
target_link_libraries(tahoma_lib PUBLIC ${EXT_LIBS})
target_include_directories(tahoma_lib PUBLIC "${CMAKE_SOURCE_DIR}/include")

#add_executable(tahoma_exe buffer.cpp ${BACKWARD_ENABLE})   # use this for debugging a piece of code
add_executable(tahoma_exe main.cpp ${BACKWARD_ENABLE})
set_target_properties(tahoma_exe PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set_target_properties(tahoma_exe PROPERTIES OUTPUT_NAME "tahoma")
target_link_libraries(tahoma_exe PUBLIC tahoma_lib)
add_backward(tahoma_exe)   # debug info if process crashes


#add_executable(mnist mnist.cpp ${BACKWARD_ENABLE})
#add_backward(mnist)   # debug info for crash
#set_target_properties(mnist PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
#target_link_libraries(mnist ${EXT_LIBS})

