#include <iostream>
#include <typeinfo>
#include <math.h>
#include <cassert>
#include <initializer_list>
#include <memory>

/**
 * @brief Test if expected and actual are equal.
 * @param expected The expected value.
 * @param actual The actual value.
 * @return True if expexted equals the actual value.
 */
bool equals(double expected, double actual)
{
    return fabs(expected - actual) < 0.0001;
}

/**
 * @brief Test the code.
 */
void test(void)
{
    //test equals
    assert(equals(2.0, 2.0));
    assert(!equals(2.0001, 2.00021));
    
    
    
    
    std::cout << "All test passed!" << std::endl;
}

template<typename T>
class Vector {
private:
    T* data;
    int size;
    
public:
public:
    /**
     * @brief Default constructor for a vector.
     * @return An vector with size 0.
     */
    Vector()
    : size(0), 
      data(nullptr)    
    { }
    
    /**
     * @brief Create a vector of size n
     * @param n the size of the vector.
     * @return A vector with size n.
     */
    Vector(int n)
    : data(new T[n]),
      size(n)
    { }

    /**
     * @brief Create a vector and initialize the vector with the values in the list.
     * @param list the values for the fector.
     * @return A vector initialized with the values from the list.
     */
    Vector(std::initializer_list<T> list)
    : Vector((int)list.size())
    {
        std::uninitialized_copy(list.begin(), list.end(), data);
    }
    
    /**
     * @brief Destroy the vector.
     * @return 
     */
    ~Vector()
    {
        size=0;
        delete[] data;
    }

};

int main(){
    test();

    return 0;
}
