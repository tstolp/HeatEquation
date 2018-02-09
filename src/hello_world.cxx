/**
 * \file hello.cxx
 *
 * This file is part of the course tw3720tu:
 * Object Oriented Scientific Programming with C++
 *
 * \author Matthias Moller
 *
 */

// Include header file for standard input/output stream library
#include <iostream>

// The global main function that is the designated start of the program
int main(){

  // Write the string 'Hello World' to the default output stream and
  // terminate with a new line (that is what std::endl does)
  std::cout << "Hello world!" << std::endl;

  // Return code 0 to the operating system (=no error)
  return 0;
}
