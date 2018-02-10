#include <iostream>
#include <typeinfo>
#include <math.h>
#include <cassert>
#include <initializer_list>
#include <memory>

template<typename T, typename S>
bool equals(const T a, const S b)
{
    return false;
}

template<typename T>
bool equals(const T a, const T b)
{
    return a == b;
}

bool equals(const double a, const double b)
{
    return fabs(a - b) < 0.0001;  
}


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
    Vector(const int n)
    : data(new T[n]),
      size(n)
    { }


    Vector(const std::initializer_list<T> list)
    : Vector((int)list.size())
    {
        std::uninitialized_copy(list.begin(), list.end(), data);
    }


    /**
     * @brief Copy constructor.
     * @param v
     * @return
     */

    Vector(const Vector<T>& v)
    : Vector(v.size)
    {
        for (int i=0; i<v.size; i++)
            data[i] = v.data[i];
    }

    /**
     * @brief Move constructor.
     * @param v
     * @return
     */
    Vector(Vector&& v)
    : data(v.data),
      size(v.size)
    {
        v.size = 0;
        v.data = nullptr;
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


    /**
     * @brief Copy assignment.
     * @param other
     */
    Vector<T>& operator=(const Vector<T>& other)
    {
        if (this != &other)
            {
                delete[] data;
                size = other.size;
                data   = new double[other.size];
                for (int i=0; i<other.size; i++)
                    data[i] = other.data[i];
            }
        return *this;
    }

    /**
     * @brief Move assignment.
     * @param other
     */
    Vector<T>& operator=(Vector<T>&& other)
    {
        if (this != &other)
            {
                delete[] data;
                size = other.size;
                data   = other.data;
                other.size = 0;
                other.data   = nullptr;
            }
        return *this;
    }


    /**
     * @brief plus operator.
     * @param other
     */
    Vector<T> operator+(const Vector<T>& other) const
    {
        Vector v(size);
        if(size != other.size)
        {
            throw "Vectors don't have the same size";
        }
        else{
            for (auto i=0; i<other.size; i++)
                    v.data[i] = data[i] + other.data[i];
        }

        std::clog << "plus operator" << std::endl;
        return v;
    }

     /**
     * @brief minus operator.
     * @param other
     */
    Vector<T> operator-(const Vector<T>& other) const
    {
        Vector v(size);
        if(size != other.size)
        {
            throw "Vectors don't have the same size";
        }
        else{
            for (auto i=0; i<other.size; i++)
                    v.data[i] = data[i] - other.data[i];
        }

        std::clog << "minus operator" << std::endl;
        return v;
    }

  /**  Vector operator*(const Vector<T>& scalar) //TODO 
    {
        Vector v(size);
        for (auto i=0; i<size; i++)
            v.data[i] = scalar*data[i];
        return v;
    }**/
    
    Vector<T> operator*(const T scalar) const
    {
        Vector v(size);
        for (auto i=0; i<size; i++)
            v.data[i] = scalar*data[i];
        return v;
    }
    
    T& operator[] (const int index) const
    {
        if (size < index || index < 0)
            throw "Index out of bounds";

        return data[index];
    }
    
    int get_size() const 
    { 
        return size; 
    }
    

    void print()
    {
        std::cout << "[";
        for(int x = 0; x < size; x++) {
            if (x != 0)
                std::cout << ", ";
            std::cout << data[x];
        }
        std::cout << "]" << std::endl;        
    }   
};

template <typename T>
bool equals(const Vector<T>& a, const Vector<T>& b)
{
    if (a.get_size() != b.get_size())
        return false;
    for(int i = 0; i < a.get_size(); i++)
    {
        if(!equals(a[i], b[i]))
            return false;
    }
       
    return true;
}
    
template <typename T>
Vector<T> operator*(const T scalar, const Vector<T> &v) {
    return v * scalar;
}



/**
* @brief Function that returns the dot product of two vectors.
* @param two vectors of the same length.
* @return the dot product.
*/

template<typename T>
T dot(const Vector<T>& l, const Vector<T>& r)
{
    if(l.get_size() != r.get_size())
    {
        throw "Vectors don't have the same size";
    }
        
    // Calculate the dot product
    double d=0;
    for (auto i=0; i<l.size; i++)
        d += l.data[i]*r.data[i];
    return d;
}

template<typename T>
class Matrix {
private:
    int size_m;
    int size_n;
    T** data;
public:
    Matrix(const int m, const int n):
     size_n(n),
     size_m(m),
     data(new T*[m])
    {
    for(int i = 0; i < m; i++)
        data[i] = new T[n];
    }
     
    T& operator[] (const std::initializer_list<int> index) {
        if (index.size() != 2)
            throw "Needs 2 elements for the index";
        //TODO check size_mn
        return data[index.begin()[0]][index.begin()[1]];
    }
    
    ~Matrix()
    {
        size_m = 0;
        size_n = 0;
        for(int i = 0; i < size_m; i++)
            delete[] data[i];
        delete[] data;
    }
    
    /**
     * @brief Copy constructor.
     * @param v
     * @return
     */

    Matrix(const Matrix<T>& m)
    : Matrix<T>(m.size_m, m.size_n)
    {
        for (int i=0; i<m.size_m; i++)
            for (int j=0; i<m.size_n; j++)
                data[i][j] = m.data[i][j];
    }

    /**
     * @brief Move constructor.
     * @param v
     * @return
     */
    Matrix(Matrix<T>&& m)
    : data(m.data),
      size_m(m.size_m),
      size_n(m.size_n)
    {
        m.size_m = 0;
        m.size_n = 0;
        m.data = nullptr;
    }
    
    /**
     * @brief Copy assignment.
     * @param other
     */
    Matrix<T>& operator=(const Matrix<T>& other)
    {
        if (this != &other)
        {
            //TODO check size
            for (int i=0; i<other.size_m; i++)
                for (int j=0; i<other.size_n; j++)
                    data[i][j] = other.data[i][j];
        }
        return *this;
    }

    /**
     * @brief Move assignment.
     * @param other
     */
    Matrix<T>& operator=(Matrix<T>&& other)
    {
        if (this != &other)
            {
                delete[] data;
                //TODO check size
                for (int i=0; i<other.size_m; i++)
                    for (int j=0; i<other.size_n; j++)
                        data[i][j] = other.data[i][j];
                data = other.data;
                
                other.size_m = 0;
                other.size_n = 0;
                other.data   = nullptr;
            }
        return *this;
    }
    
    Vector<T> matvec(const Vector<T> &v)
    {
        Vector<T> result(v.size);
        //TODO check sizes;
        for (int i=0; i<v.size_m; i++)
            for (int j=0; i<v.size_n; j++)
                result[i] += data[i][j] * v[j];
        
    }

    Vector<T> operator*(const Vector<T> &v)
    {
        return matvec(v);
    }
    
    void print()
    {
        for(int y = 0; y < size_m; y++) {
            std::cout << "[";
            for(int x = 0; x < size_n; x++) {
                if (x != 0)
                    std::cout << ", ";
                std::cout << data[y][x];
            }
            std::cout << "]" << std::endl;
        }
        
    }
 
};

template<typename T>
int cg(
    const Matrix<T> &A, const Vector<T> &b, Vector<T> &x, T tol, int maxiter)
{
    Vector<T> r = b - A * x;
    Vector<T> p = r;
    int result = -1;
    for(int k = 0; k < maxiter; k++)
    {
        T alpha = dot(r, r) / dot(A * p, p);
        Vector<T> x_n = x + alpha * p;
        Vector<T> r_n = r - alpha * A * p;
        if (dot(r_n, r_n) < tol*tol) {
            result = k;
            x = x_n;
            
        }
        T beta = dot(r_n, r_n) / dot(r, r);
        p = r_n + beta * p;
        r = r_n;
        x = x_n;
    }
    return result;
}

class Heat1D
{
private:
    Matrix<double> M;
    
public:
    Heat1D(double alpha, int m, double dt) :
        M(m,m)
    {
        for (int i = 0; i < m; i++)
        {
            int l = i - 1;
            int r = i + 1;
            double dx = 1.0/(m+1);
            double s = alpha * dt / (dx * dx);
            M[{i,i}] = 1 + 2 * s;
            if (l >= 0)
                M[{i,l}] = -1 * s;
            if (r < m)
                M[{i,r}] = -1 * s;
        }
        
    }
};

void test_vector_constructor(void)
{
    Vector<double> A;
    Vector<double> B(0);
    Vector<double> C({});
    Vector<int> E;
    Vector<double> F(1);
    Vector<double> G({1});
    Vector<double> H({1,2,3});
    Vector<double> I({1,2,3});
    Vector<double> J({1,2,3.1});
    Vector<double> K({0.9,2,3});
    Vector<double> L = I;
    assert(A.get_size() == 0);
    assert(B.get_size() == 0);
    assert(C.get_size() == 0);
    assert(F.get_size() == 1);
    assert(H.get_size() == 3);
    assert(equals(A,B));
    assert(equals(A,C));
    assert(!equals(A,E));
    
    assert(!equals(A,F));
    assert(!equals(A,G));
    assert(!equals(F,G));
    assert(!equals(G,H));
    assert(equals(H,I));
    assert(!equals(I,J));
    assert(!equals(I,K));
    assert(equals(L,I));
    assert(&L != &I);
    I[0] = 0.9;
    assert(!equals(L,I));
    assert(equals(I,K));
    // std::cout << dot(H, I) << std::endl;
}

void test_vector_operations(void)
{
    Vector<double> A({1,2,3});
    Vector<double> B({2,4,-1});
    
    Vector<double> expected({3,6,2});
    assert(equals(expected, A+B));
    
    expected = Vector<double>({-1,-2,4});
    assert(equals(expected, A-B));
    
    expected = Vector<double>({4, 8, -2});
    assert(equals(expected, 2.0 * B));
    assert(equals(expected, B * 2.0));
    
}

void test_matrix(void)
{
    Matrix<double> test(3,4);
    test[{1,2}] = 2.0;
    test.print();
}

/**
 * @brief Test the code.
 */
void test(void)
{
    test_vector_constructor();
    test_vector_operations();
    test_matrix();
    std::cout << "All test passed!" << std::endl;
}

int main(){
    test();

    return 0;
}
