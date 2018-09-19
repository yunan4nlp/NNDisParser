include_directories(
  model
)

add_executable(NNDisParser-synmlp-do NNDisParser.cpp)
target_link_libraries(NNDisParser-synmlp-do ${LIBS})
