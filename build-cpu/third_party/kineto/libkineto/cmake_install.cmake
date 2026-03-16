# Install script for directory: /data/pytorch-source/third_party/kineto/libkineto

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/data/pytorch-source/torch")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/data/pytorch-source/build/third_party/kineto/libkineto/third_party/dynolog/dynolog/src/ipcfabric/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY MESSAGE_NEVER FILES "/data/pytorch-source/build/lib/libkineto.a")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/data/pytorch-source/build/third_party/kineto/libkineto/CMakeFiles/kineto.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/kineto" TYPE FILE MESSAGE_NEVER FILES
    "/data/pytorch-source/third_party/kineto/libkineto/include/AbstractConfig.h"
    "/data/pytorch-source/third_party/kineto/libkineto/include/ActivityProfilerInterface.h"
    "/data/pytorch-source/third_party/kineto/libkineto/include/ActivityTraceInterface.h"
    "/data/pytorch-source/third_party/kineto/libkineto/include/ActivityType.h"
    "/data/pytorch-source/third_party/kineto/libkineto/include/Config.h"
    "/data/pytorch-source/third_party/kineto/libkineto/include/ClientInterface.h"
    "/data/pytorch-source/third_party/kineto/libkineto/include/GenericTraceActivity.h"
    "/data/pytorch-source/third_party/kineto/libkineto/include/IActivityProfiler.h"
    "/data/pytorch-source/third_party/kineto/libkineto/include/ILoggerObserver.h"
    "/data/pytorch-source/third_party/kineto/libkineto/include/ITraceActivity.h"
    "/data/pytorch-source/third_party/kineto/libkineto/include/LoggingAPI.h"
    "/data/pytorch-source/third_party/kineto/libkineto/include/TraceSpan.h"
    "/data/pytorch-source/third_party/kineto/libkineto/include/ThreadUtil.h"
    "/data/pytorch-source/third_party/kineto/libkineto/include/libkineto.h"
    "/data/pytorch-source/third_party/kineto/libkineto/include/time_since_epoch.h"
    "/data/pytorch-source/third_party/kineto/libkineto/include/output_base.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/kineto/kinetoLibraryConfig.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/kineto/kinetoLibraryConfig.cmake"
         "/data/pytorch-source/build/third_party/kineto/libkineto/CMakeFiles/Export/12060e67814ad64b001bb7818b5234f4/kinetoLibraryConfig.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/kineto/kinetoLibraryConfig-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/kineto/kinetoLibraryConfig.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/cmake/kineto" TYPE FILE MESSAGE_NEVER FILES "/data/pytorch-source/build/third_party/kineto/libkineto/CMakeFiles/Export/12060e67814ad64b001bb7818b5234f4/kinetoLibraryConfig.cmake")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/cmake/kineto" TYPE FILE MESSAGE_NEVER FILES "/data/pytorch-source/build/third_party/kineto/libkineto/CMakeFiles/Export/12060e67814ad64b001bb7818b5234f4/kinetoLibraryConfig-release.cmake")
  endif()
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/data/pytorch-source/build/third_party/kineto/libkineto/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
