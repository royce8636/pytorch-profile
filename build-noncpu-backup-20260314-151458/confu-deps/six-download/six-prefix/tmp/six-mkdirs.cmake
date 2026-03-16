# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/data/pytorch-source/build/confu-srcs/six"
  "/data/pytorch-source/build/confu-deps/six"
  "/data/pytorch-source/build/confu-deps/six-download/six-prefix"
  "/data/pytorch-source/build/confu-deps/six-download/six-prefix/tmp"
  "/data/pytorch-source/build/confu-deps/six-download/six-prefix/src/six-stamp"
  "/data/pytorch-source/build/confu-deps/six-download/six-prefix/src"
  "/data/pytorch-source/build/confu-deps/six-download/six-prefix/src/six-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/data/pytorch-source/build/confu-deps/six-download/six-prefix/src/six-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/data/pytorch-source/build/confu-deps/six-download/six-prefix/src/six-stamp${cfgdir}") # cfgdir has leading slash
endif()
