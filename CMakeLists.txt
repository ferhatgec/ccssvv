cmake_minimum_required(VERSION 3.30)

# import std; UUIDs
if (CMAKE_VERSION VERSION_GREATER_EQUAL "4.4.0")
  set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "f35a9ac6-8463-4d38-8eec-5d6008153e7d")
elseif (CMAKE_VERSION VERSION_GREATER_EQUAL "4.3.0")
  set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "451f2fe2-a8a2-47c3-bc32-94786d8fc91b")
elseif (CMAKE_VERSION VERSION_GREATER_EQUAL "4.1.0")
  set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "d0edc3af-4c50-42ea-a356-e2862fe7a444")
elseif (CMAKE_VERSION VERSION_GREATER_EQUAL "4.0.0")
  set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "a9e1cf81-9932-4810-974b-6eccaf14e457")
else ()
  # 3.30/3.31
  set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "0e5b6991-d74f-4b3d-a41c-cf096e0b2508")
endif ()

project(csv
  VERSION 1.0.0
  DESCRIPTION "C++23 CSV parser"
  LANGUAGES CXX
)

option(CSV_BUILD_EXAMPLES "Build example code" ${PROJECT_IS_TOP_LEVEL})
option(CSV_ENABLE_AVX2 "AVX2" ON)
option(CSV_ENABLE_NATIVE "-march=native" OFF)

add_library(csv STATIC)
add_library(csv::csv ALIAS csv)

target_compile_features(csv PUBLIC cxx_std_23)

set_target_properties(csv PROPERTIES
  CXX_STANDARD 23
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  CXX_MODULE_STD ON
)

target_sources(csv
  PUBLIC
  FILE_SET CXX_MODULES
  BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
  FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/csv.cppm
)

if (CSV_ENABLE_NATIVE)
  if (MSVC)
    target_compile_options(csv PUBLIC /arch:AVX2)
  else ()
    target_compile_options(csv PUBLIC -march=native)
  endif ()
elseif (CSV_ENABLE_AVX2)
  if (MSVC)
    target_compile_options(csv PUBLIC /arch:AVX2)
  elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    target_compile_options(csv PUBLIC -mavx2 -mbmi)
  endif ()
endif ()

if (MSVC)
  target_compile_options(csv PRIVATE /W4 /permissive-)
else ()
  target_compile_options(csv PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
endif ()

if (CSV_BUILD_EXAMPLES)
  add_executable(csv_example example.cpp)
  set_target_properties(csv_example PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    CXX_MODULE_STD ON
  )
  target_link_libraries(csv_example PRIVATE csv::csv)
endif ()