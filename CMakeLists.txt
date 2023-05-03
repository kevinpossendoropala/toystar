cmake_minimum_required(VERSION 3.16)
project(Cabana_HelloWorld)
if(NOT TARGET Cabana::cabanacore)
  find_package(Cabana)
endif()

add_executable(teste teste_1.cpp)
target_link_libraries(teste Cabana::cabanacore)
