#include <iostream>
#include <typeinfo>
#include <math.h>
#include <cassert>
#include <initializer_list>
#include <memory>


template<typename T>
class Vector {
private:
    T* data;
    int size;
    
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
    
    Vector(const Vector<T>& v)
    : Vector(v.size)
    {
        for (int i=0; i<v.size; i++)
            data[i] = v.data[i];
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
    

    template<typename A>
    bool equals(Vector<A> other)
    {        
        return false;
    }
    
    bool equals(Vector<T> other)
    {
        std::cout << "1" << std::endl;
        if (this->size != other.size)
            return false;
        std::cout << "2" << std::endl;
        for(int i = 0; i < this->size; i++)
        {
            std::cout << this->data[i] << " " << other.data[i] << std::endl;
            if(!fabs(this->data[i] - other.data[i]) < 0.0001)
                return false;
        }
        
        return true;
    }
    

    

};

/**
 * @brief Test the code.
 */
void test(void)
{
    //test equals
    
    Vector<double> A;
    Vector<double> B(0);
    Vector<double> C({});
    Vector<int> E;
    Vector<double> F(1);
    Vector<double> G({1});
    Vector<double> H({1,2,3});
    Vector<double> I({1,2,3});
    Vector<double> J({1,2,0});
    Vector<double> K({0,2,3});
    assert(A.equals(B));
    assert(A.equals(C));
    assert(!A.equals(E));
    assert(!A.equals(F));
    assert(!A.equals(G));
    assert(!F.equals(G));
    assert(!G.equals(H));
    std::cout << "test" << std::endl;
    assert(H.equals(I));
    assert(!I.equals(J));
    assert(!I.equals(K));
    
    std::cout << "All test passed!" << std::endl;
}

int main(){
    test();

    return 0;
}
